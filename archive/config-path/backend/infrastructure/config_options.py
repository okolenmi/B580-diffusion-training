"""PydanticConfigOptions -- ConfigOptions: schema + UI metadata, merged.

A pure function of the config model and ``config_ui_data``: no config
file is read, no filesystem is touched, so the result is fetchable
once and reused across every config (field *values* come from
``ConfigFiles.read``).

Output: a flat list -- synthetic per-launch options first (they own
group "0. LAUNCH"), then one entry per schema field with any
hand-authored overrides merged in. Insertion order is render order
within a group.
"""

from __future__ import annotations

from typing import Any

from ..application.ports.config_options import ConfigOptions
from .config_schema import build_schema_options
from .config_ui_data import EXTRAS, SYNTHETIC_OPTIONS

# start_from / reset_optimizer are real TrainingConfig fields (so the
# schema picks them up automatically), but the UI must only show the
# per-launch SYNTHETIC_OPTIONS versions -- excluded here so they never
# appear twice.
_EXCLUDE_FROM_SCHEMA = {"start_from", "reset_optimizer"}


def _humanize(dotted_key: str) -> str:
    name = dotted_key.rsplit(".", 1)[-1]
    return name.replace("_", " ").title()


def _naive_choice_label(value: Any) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


class PydanticConfigOptions(ConfigOptions):
    def schema(self) -> list[dict[str, Any]]:
        schema_opts = build_schema_options()
        options: list[dict[str, Any]] = [dict(opt) for opt in SYNTHETIC_OPTIONS]

        for dotted_key, base in schema_opts.items():
            if dotted_key in _EXCLUDE_FROM_SCHEMA:
                continue
            extra = EXTRAS.get(dotted_key, {})

            opt: dict[str, Any] = {
                "id": dotted_key,
                "label": extra.get("label", _humanize(dotted_key)),
                "type": base["type"],
                "default": base.get("default"),
                "group": extra.get("group", "General"),
            }

            if "choices" in base:
                choice_labels = extra.get("choice_labels", {})
                choice_order = extra.get("choice_order")
                raw_choices = base["choices"]
                if choice_order:
                    # Preserve a curated display order; anything not
                    # explicitly ordered is appended at the end (so a new
                    # Union variant added later still shows up instead of
                    # silently disappearing).
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

            if "file_kind" in extra:
                opt["file_kind"] = extra["file_kind"]
            if "order" in extra:
                opt["order"] = extra["order"]
            if "subgroup" in extra:
                opt["subgroup"] = extra["subgroup"]

            visible_when = dict(base.get("visible_when") or {})
            visible_when.update(extra.get("extra_visible_when") or {})
            if visible_when:
                opt["visible_when"] = visible_when

            # Drop Nones, but never the default key itself (a null default
            # is meaningful: "no value" differs from "no default").
            opt = {k: v for k, v in opt.items() if v is not None or k == "default"}
            options.append(opt)

        return options
