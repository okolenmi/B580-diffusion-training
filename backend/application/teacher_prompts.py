"""Teacher prompt configs -- validate + assemble for ``generate_teacher``.

One pure-stdlib module shared by two processes that must never
disagree about what a legal teacher launch is:

* ``StartDatasetTask`` calls ``teacher_payload`` *before* the task row
  exists, so a bad range or an empty prompt list surfaces as the error
  envelope (``invalid_query``) instead of a failed child;
* ``dataset_task_worker`` calls ``build_pos_cfg``/``build_neg_cfg`` to
  re-assemble the exact configs ``manager.builder.DataTaskRunner``
  expects (list-of-strings | keywords dict | single string).

Validation lives here rather than in the DTO (a wire-shaped dataclass)
or the worker (too late -- the row already exists). Every failure is a
``ValueError``; the use case maps it to ``InvalidQueryError``.

The assembled configs themselves are deliberately *not* stored in the
task ``params`` record: the flat fields are the launch payload (the UI
and any audit read those), and the configs are derived -- rebuilding
them is cheap and free of ambiguity.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dto import TeacherTaskParams

# Kept in sync with core/noise_schedule.sample_timestep (T_MODES) --
# the builder feeds non-uniform modes through it for trajectory spacing.
T_MODES: tuple[str, ...] = ("uniform", "low", "mid", "high", "logit")
PROMPT_MODES: tuple[str, ...] = ("list", "keywords")
MODEL_TYPES: tuple[str, ...] = ("eps", "vpred")

# Numeric bounds: wide enough for every legacy default, narrow enough
# to catch fat-fingered launches before they spawn a child.
_CFG_RANGE = (0.0, 100.0)
_STEPS_RANGE = (1, 500)
_T_RANGE = (0, 1000)
_N_CONDITIONS = (1, 10_000)
_N_SAMPLES = (1, 1_000)
_BATCH = (1, 256)
_LATENT = (8, 512)


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").split("\n") if ln.strip()]


def _keywords_dict(
    *,
    keywords: str,
    keywords_file: str,
    template: str,
    min_keywords: int,
    max_keywords: int,
    side: str,
) -> dict:
    """Assemble the builder's keywords dict (see manager/builder.py
    ``_prepare_keywords``), refusing configurations that would
    silently generate empty prompts: keywords mode with neither a
    keyword pool nor a readable file, an unreadable file, or an
    inverted min/max window."""
    pool = _lines(keywords)
    file_path = Path(keywords_file) if keywords_file else None
    if not pool and file_path is None:
        raise ValueError(
            f"{side}: keywords mode needs a keyword list or keywords_file"
        )
    if file_path is not None:
        # Same posture as image_dir: server-side paths are allowed
        # anywhere, but they must exist -- the builder skips a missing
        # file silently, which would turn this launch into a wall of
        # empty captions.
        if not file_path.is_file():
            raise ValueError(f"{side}: keywords_file not found: {keywords_file!r}")
        if file_path.stat().st_size == 0:
            raise ValueError(f"{side}: keywords_file is empty: {keywords_file!r}")
    if not 1 <= min_keywords <= max_keywords:
        raise ValueError(
            f"{side}: need 1 <= min_keywords <= max_keywords, "
            f"got {min_keywords}..{max_keywords}"
        )
    return {
        "keywords": pool or None,
        "keywords_file": keywords_file or None,
        "template": template or None,
        "min": min_keywords,
        "max": max_keywords,
    }


def build_pos_cfg(t: TeacherTaskParams) -> list[str] | dict:
    """Positive-prompt config for the builder (``_generate_prompts``)."""
    if t.prompt_mode == "list":
        prompts = _lines(t.prompts)
        if not prompts:
            raise ValueError(
                "prompt_mode 'list' needs at least one non-empty line in 'prompts'"
            )
        return prompts
    if t.prompt_mode == "keywords":
        return _keywords_dict(
            keywords=t.keywords,
            keywords_file=t.keywords_file,
            template=t.template,
            min_keywords=t.min_keywords,
            max_keywords=t.max_keywords,
            side="prompts",
        )
    raise ValueError(
        f"unknown prompt_mode {t.prompt_mode!r}; "
        f"expected one of {list(PROMPT_MODES)}"
    )


def build_neg_cfg(t: TeacherTaskParams) -> str | dict:
    """Negative-prompt config. mode='list' is a single string applied
    to every sample and may legitimately be ``""`` (no negative)."""
    if t.neg_mode == "list":
        return t.negative_prompt
    if t.neg_mode == "keywords":
        return _keywords_dict(
            keywords=t.neg_keywords,
            keywords_file=t.neg_keywords_file,
            template=t.neg_template,
            min_keywords=t.neg_min_keywords,
            max_keywords=t.neg_max_keywords,
            side="negative prompts",
        )
    raise ValueError(
        f"unknown neg_mode {t.neg_mode!r}; expected one of {list(PROMPT_MODES)}"
    )


def _in(name: str, value: int | float, bounds: tuple, *, integer: bool = True) -> None:
    lo, hi = bounds
    if not lo <= value <= hi:
        kind = "int" if integer else "number"
        raise ValueError(f"{name} must be a {kind} in {lo}..{hi}, got {value}")


def validate_teacher(t: TeacherTaskParams) -> None:
    """Range/mode checks. Prompt content is validated by the builders
    (they are the ones who would otherwise produce nothing)."""
    _in("cfg_min", t.cfg_min, _CFG_RANGE, integer=False)
    _in("cfg_max", t.cfg_max, _CFG_RANGE, integer=False)
    if t.cfg_min > t.cfg_max:
        raise ValueError(
            f"cfg_min must be <= cfg_max, got {t.cfg_min}..{t.cfg_max}"
        )
    _in("steps_min", t.steps_min, _STEPS_RANGE)
    _in("steps_max", t.steps_max, _STEPS_RANGE)
    if t.steps_min > t.steps_max:
        raise ValueError(
            f"steps_min must be <= steps_max, got {t.steps_min}..{t.steps_max}"
        )
    _in("t_low", t.t_low, _T_RANGE)
    _in("t_high", t.t_high, _T_RANGE)
    if t.t_low > t.t_high:
        raise ValueError(f"t_low must be <= t_high, got {t.t_low}..{t.t_high}")
    if t.t_mode not in T_MODES:
        raise ValueError(
            f"unknown t_mode {t.t_mode!r}; expected one of {list(T_MODES)}"
        )
    _in("n_conditions", t.n_conditions, _N_CONDITIONS)
    _in("n_samples_per_cond", t.n_samples_per_cond, _N_SAMPLES)
    _in("batch_size", t.batch_size, _BATCH)
    _in("latent_size", t.latent_size, _LATENT)
    if t.model_type not in MODEL_TYPES:
        raise ValueError(
            f"unknown model_type {t.model_type!r}; "
            f"expected one of {list(MODEL_TYPES)}"
        )
    # neg_mode is checked by build_neg_cfg (mode validity is content
    # validation there too); call both so a single entry point works.
    build_pos_cfg(t)
    build_neg_cfg(t)


def teacher_payload(t: TeacherTaskParams) -> dict[str, Any]:
    """Validated flat ``params`` record for the task row.

    Raises ``ValueError`` on anything the builder could not execute
    meaningfully; the use case turns that into ``invalid_query``.
    Running the builders here (not just the range checks) is what
    makes "empty prompts" fail *before* the child spawns.
    """
    validate_teacher(t)
    return asdict(t)
