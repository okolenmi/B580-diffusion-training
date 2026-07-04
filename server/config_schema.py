"""Derives base UI option metadata directly from core.config_model's Pydantic
models, via runtime introspection (model_fields, constraint metadata, Union
members) -- not by hand-duplicating it.

This is deliberately kept in server/, not core/: core/config_model.py is
used directly by CLI-only users who never touch the web UI, and shouldn't
carry any UI-specific baggage (labels, groups, help text formatted for a
web form). This module is the bridge -- it reads the schema, but doesn't
modify or annotate core/ at all.

What's auto-derived here (and therefore CANNOT drift from the real schema):
  - field path (dotted, e.g. "common.steps")
  - type: text / number / checkbox / select
  - default value
  - min / max (from Field(ge=..., le=...) constraints)
  - choices (raw values, from Literal[...] members)
  - a base visible_when, for fields that only exist on one variant of a
    discriminated union (e.g. tuning.rank only exists on LoRATuning, so it
    automatically gets visible_when={"tuning.method": "lora"})

What's explicitly NOT here (see config_ui.py for that layer):
  - display labels, help text overrides, UI grouping
  - friendly choice labels ("fused-adafactor" -> "Fused Adafactor")
  - additional visible_when conditions beyond "which union variant is this"
    (e.g. cache fields also depend on common.data_source == "teacher",
    which is cross-cutting business logic the schema has no way to know)
"""

from __future__ import annotations

import typing
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from core.config_model import TrainingConfig

SchemaOption = dict[str, Any]


def _is_basemodel(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """Return (inner_type, was_optional) for Optional[X] / Union[X, None]."""
    if get_origin(tp) is typing.Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and type(None) in get_args(tp):
            return args[0], True
    return tp, False


def _extract_ge_le(field_info: FieldInfo) -> tuple[float | None, float | None]:
    lo = hi = None
    for constraint in field_info.metadata:
        if hasattr(constraint, "ge"):
            lo = constraint.ge
        elif hasattr(constraint, "gt"):
            lo = constraint.gt
        if hasattr(constraint, "le"):
            hi = constraint.le
        elif hasattr(constraint, "lt"):
            hi = constraint.lt
    return lo, hi


def _default_value(field_info: FieldInfo) -> Any:
    if field_info.default is not PydanticUndefined:
        return field_info.default
    if field_info.default_factory is not None:
        try:
            return field_info.default_factory()
        except Exception:
            return None
    return None


def _leaf_option(field_info: FieldInfo, tp: Any) -> SchemaOption | None:
    """Build the {type, default, min, max, choices} portion for a scalar
    (non-BaseModel, non-Union-of-BaseModel) field. Returns None for types
    this generator doesn't know how to render (falls back to hand-authored
    config_ui.py entries, if any, or is simply omitted)."""
    tp, _ = _unwrap_optional(tp)
    origin = get_origin(tp)

    opt: SchemaOption = {"default": _default_value(field_info)}

    if origin is typing.Literal:
        opt["type"] = "select"
        opt["choices"] = list(get_args(tp))
        return opt
    if tp is bool:
        opt["type"] = "checkbox"
        return opt
    if tp in (int, float):
        opt["type"] = "number"
        lo, hi = _extract_ge_le(field_info)
        if lo is not None:
            opt["min"] = lo
        if hi is not None:
            opt["max"] = hi
        return opt
    if tp is str:
        opt["type"] = "text"
        return opt
    return None


def _union_members(tp: Any) -> list[type[BaseModel]] | None:
    if get_origin(tp) is typing.Union:
        args = [a for a in get_args(tp) if _is_basemodel(a)]
        if args:
            return args
    return None


def _discriminator_field_name(member: type[BaseModel]) -> str | None:
    """Find the Literal-typed field that discriminates this union member
    (e.g. "method" on LoRATuning, "mode" on TrajectoryCache)."""
    for name, info in member.model_fields.items():
        if get_origin(info.annotation) is typing.Literal:
            return name
    return None


def _walk(model: type[BaseModel], prefix: str, base_visible_when: dict | None,
          out: dict[str, SchemaOption]):
    for name, field_info in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        tp, _ = _unwrap_optional(field_info.annotation)

        union_members = _union_members(tp)
        if union_members is not None:
            # Discriminated union (tuning: TuningMethod, cache: CacheConfig).
            # The discriminator field itself (e.g. tuning.method) becomes
            # its own select control; the *other* fields on each variant
            # get folded into the flat dotted-path space, auto-tagged with
            # visible_when={that path: that variant's discriminator value}.
            # The discriminator control itself (e.g. "tuning.method") goes
            # in FIRST, before any of the mode-specific fields below it --
            # both because that's the sensible reading order (pick the mode
            # before configuring it) and because insertion order here
            # becomes render order in the frontend (option-tree.js renders
            # each group's options in the order they appear in the flat
            # list). Default comes from the *actual* field default
            # (default_factory), not "whichever variant happens to be
            # listed first in the Union" -- those aren't the same thing
            # (e.g. TrainingConfig.tuning defaults to DistillationTuning
            # even though LoRATuning is listed first).
            disc_values = [
                get_args(member.model_fields[_discriminator_field_name(member)].annotation)[0]
                for member in union_members
                if _discriminator_field_name(member) is not None
            ]
            if disc_values:
                first_member = union_members[0]
                disc_name = _discriminator_field_name(first_member)
                disc_path = f"{path}.{disc_name}"
                actual_default_instance = _default_value(field_info)
                actual_default = (
                    getattr(actual_default_instance, disc_name, disc_values[0])
                    if actual_default_instance is not None else disc_values[0]
                )
                out[disc_path] = {
                    "type": "select",
                    "choices": list(disc_values),
                    "default": actual_default,
                    **({"visible_when": base_visible_when} if base_visible_when else {}),
                }

            for member in union_members:
                disc_name = _discriminator_field_name(member)
                if disc_name is None:
                    continue
                disc_info = member.model_fields[disc_name]
                disc_value = get_args(disc_info.annotation)[0]
                disc_path = f"{path}.{disc_name}"

                member_visible = {disc_path: disc_value}
                if base_visible_when:
                    member_visible = {**base_visible_when, **member_visible}

                for sub_name, sub_info in member.model_fields.items():
                    if sub_name == disc_name:
                        continue
                    sub_path = f"{path}.{sub_name}"
                    sub_tp, _ = _unwrap_optional(sub_info.annotation)
                    if _is_basemodel(sub_tp) or _union_members(sub_tp) is not None:
                        # Not currently needed (no nested submodel/union
                        # inside a tuning/cache variant), but handle it
                        # instead of silently dropping fields if it's ever
                        # added.
                        _walk(sub_tp, sub_path, member_visible, out)
                        continue
                    leaf = _leaf_option(sub_info, sub_tp)
                    if leaf is None:
                        continue
                    leaf["visible_when"] = member_visible
                    out[sub_path] = leaf
            continue

        if _is_basemodel(tp):
            _walk(tp, path, base_visible_when, out)
            continue

        leaf = _leaf_option(field_info, tp)
        if leaf is None:
            continue
        if base_visible_when:
            leaf["visible_when"] = base_visible_when
        out[path] = leaf


def build_schema_options() -> dict[str, SchemaOption]:
    """Introspect TrainingConfig and return {dotted_path: SchemaOption}."""
    out: dict[str, SchemaOption] = {}
    _walk(TrainingConfig, "", None, out)
    return out
