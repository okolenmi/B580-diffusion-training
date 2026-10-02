"""Settings-aware path resolution -- one policy, two consumers.

``WorkspaceLayout`` (what launch/artifacts treat as real) and
``SqliteSettingsStore`` (what the API reports as resolved) must
agree exactly; both delegate to the functions here so the tiers
cannot drift apart.

Tier order (deliberate):

* ``comfy_dir`` / ``venv_python``: environment and workspace
  conventions first, persisted override last -- a freshly edited
  .env must win over a stale value somebody set through the UI
  months ago.
* ``checkpoints_dir`` / ``loras_dir``: persisted override first --
  that setting exists precisely to point somewhere other than the
  auto-detected layout.
* ``models_dir``: one key for the whole model tree, consulted only when
  neither of the two specific overrides is set. It is the knob for
  "this app does not live inside a ComfyUI checkout" -- the default
  arrangement is ComfyUI's own layout, and this is where that default
  gets changed rather than hard-coded.

``comfy_dir`` is the only resolver that can fail (it raises
``RuntimeError`` when nothing identifies a ComfyUI install); every
other tier falls through to a working default. ``get_setting`` never
raises (the store's ``get`` swallows storage failures -- a broken
database must not be able to crash path resolution).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections.abc import Callable

GetSetting = Callable[[str, str], str]


def import_paths(project_root: Path):
    """Import the repo's ``paths`` module (its documented .env
    fill-on-import runs here, once, never as a backend import side
    effect)."""
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import paths  # noqa: PLC0415 -- explicit repo bridge

    return paths


def comfy_dir(project_root: Path, get_setting: GetSetting) -> Path:
    paths = import_paths(project_root)
    try:
        return paths.get_comfy_dir()
    except RuntimeError:
        override = get_setting("comfy_dir", "")
        if override and Path(override).is_dir():
            return Path(override)
        raise


def venv_python(project_root: Path, get_setting: GetSetting) -> str:
    import_paths(project_root)  # load repo .env before reading env
    # A *configured* interpreter is used as-is, never silently
    # substituted: a stale/bogus value fails loudly at spawn instead of
    # the trainer quietly running under a different Python than asked.
    env = os.environ.get("VENV_PYTHON", "")
    if env:
        return env
    sibling = project_root.parent / "venv" / "bin" / "python"
    if sibling.exists():
        return str(sibling)
    override = get_setting("venv_python", "")
    if override:
        return override
    return "python"


def models_dir(
    # project_root is unused: the setting names an absolute path, so
    # there is nothing to resolve it against. Kept anyway because every
    # resolver here takes (project_root, get_setting) and the one function
    # with a different shape is the one a caller has to check. It would be
    # read if the unset path ever gained a convention-based fallback.
    project_root: Path,  # noqa: ARG001 -- see above
    get_setting: GetSetting,
) -> Path | None:
    """The single knob for "model files live over here", or None.

    One key rather than the three below it, because the relationship
    between them is a property of *ComfyUI's* layout, not of this app --
    and borrowing another project's layout is exactly the assumption that
    stops holding if the models ever move (docs 08 Q7). Point this at a
    directory and checkpoints/loras become ``<root>/checkpoints`` and
    ``<root>/loras``.

    None rather than a default: the caller decides what "unset" means.
    In ``checkpoints_dir`` it means "ask ComfyUI".
    """
    override = get_setting("models_dir", "")
    if override and Path(override).is_dir():
        return Path(override)
    return None


def checkpoints_dir(project_root: Path, get_setting: GetSetting) -> Path:
    # Order is deliberate and unchanged in spirit: the specific override
    # wins over the coarse one, and the coarse one wins over ComfyUI's
    # auto-detection. A stored `checkpoints_dir` is an explicit choice
    # about that directory; `models_dir` is an explicit choice about
    # where the model tree is.
    override = get_setting("checkpoints_dir", "")
    if override and Path(override).is_dir():
        return Path(override)
    models = models_dir(project_root, get_setting)
    if models is not None:
        return models / "checkpoints"
    return import_paths(project_root).get_checkpoints_dir()


def loras_dir(project_root: Path, get_setting: GetSetting) -> Path:
    override = get_setting("loras_dir", "")
    if override and Path(override).is_dir():
        return Path(override)
    models = models_dir(project_root, get_setting)
    if models is not None:
        return models / "loras"
    return import_paths(project_root).get_loras_dir()


def datasets_dir(project_root: Path) -> Path:
    """No override tier: managed datasets live beside the project."""
    return project_root / "datasets"
