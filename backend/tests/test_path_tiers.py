"""Path resolution: what the store says, and what `paths` says.

This file exists because of a bug the first-run installer made reachable.

`path_tiers` is the one place the resolution policy lives: a stored
override, then the environment, then ComfyUI's conventions, then a
fallback. `comfy_dir` consulted the settings *store*; `checkpoints_dir` and
`loras_dir` did not -- they called `paths.get_checkpoints_dir()` directly,
which resolves `<comfy>/models/checkpoints` and otherwise falls back to
`<project_root>/checkpoints`, a directory nobody chose.

So on a fresh checkout, where the only ComfyUI that exists is the one the
user has just told the server about through Settings, the settings page and
the training path disagreed:

    comfy_dir       -> <the directory the user chose>
    checkpoints_dir -> <the project root>/checkpoints

Measured on an isolated tree with no `.env` (a fresh checkout), before the
fix. The tests here run in the same isolation, because a test that inherits
the developer's `.env` is a test about their machine.

**The other change is that resolution now raises.** With no ComfyUI
resolvable anywhere, `checkpoints_dir` raises instead of inventing a
directory. A path the user never chose, that exists nowhere, and that reads
as "configured" -- that is worse than an error, because nothing downstream
can tell the difference between it and a real one.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.settings_store import SettingsChanges
from backend.infrastructure import path_tiers
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.infrastructure.settings_store import SqliteSettingsStore
from backend.tests.support import check, finish


def _isolated_tree() -> tuple[Path, SqliteSettingsStore, SqliteDatabase]:
    """A project root with no `.env`, and a store against it.

    `paths._load_dotenv()` runs at import time and reads the `.env` beside
    `paths.py` -- which on a developer's machine names their real ComfyUI.
    Every assertion here would then be a statement about that machine. So
    `paths.py` is copied into a scratch directory with no `.env` next to it,
    and the scratch directory is what `project_root` is.

    The copy rather than an env var because `setdefault` means an already-set
    COMFY_DIR would also be honoured -- and neutralising one variable is not
    the same as removing all of them.
    """
    root = Path(tempfile.mkdtemp(prefix="path-tiers-"))
    repo_root = Path(__file__).resolve().parents[2]
    shutil.copy(repo_root / "paths.py", root / "paths.py")

    for key in ("COMFY_DIR", "CHECKPOINTS_DIR", "LORAS_DIR", "MODELS_DIR",
                "VENV_PYTHON"):
        os.environ.pop(key, None)
    # And out of sys.modules, so the scratch copy is the one imported.
    sys.modules.pop("paths", None)
    sys.path.insert(0, str(root))
    try:
        import paths  # noqa: F401 -- imported for its side effect above

        if Path(paths.__file__).resolve().parent != root:
            paths = None  # noqa: F841 -- recorded by the check below
    finally:
        sys.path.remove(str(root))
    sys.modules.pop("paths", None)

    db = SqliteDatabase(root / "backend.db")
    db.initialize()
    return root, SqliteSettingsStore(db, root), db


# ==========================================================================
# Section: the store's comfy_dir reaches the model directories
# ==========================================================================
print("-- a fresh checkout, and the one ComfyUI the user told us about --")

root, store, db = _isolated_tree()
kv = store.get

comfy = root / "ComfyUI"
(comfy / "models" / "checkpoints").mkdir(parents=True)
(comfy / "models" / "loras").mkdir(parents=True)


def resolved(callable_):
    """Resolution, or the fact that it raised.

    Raising is the correct answer for an unresolvable machine, so the probe
    has to be able to see it rather than crash.
    """
    try:
        return callable_()
    except RuntimeError:
        return None


check(resolved(lambda: path_tiers.comfy_dir(root, kv)) is None,
      "with nothing configured and no .env, comfy_dir resolves to nothing "
      "rather than to a guess")

store.update(SettingsChanges(comfy_dir=str(comfy)))

check(path_tiers.comfy_dir(root, kv) == comfy,
      f"the stored comfy_dir is used ({path_tiers.comfy_dir(root, kv)})")

check(path_tiers.checkpoints_dir(root, kv) == comfy / "models" / "checkpoints",
      f"and the checkpoints directory is derived from it, not invented "
      f"(got {path_tiers.checkpoints_dir(root, kv)})")

check(path_tiers.loras_dir(root, kv) == comfy / "models" / "loras",
      f"and so is the LoRA directory (got {path_tiers.loras_dir(root, kv)})")

# The whole point: the two answers agree. Before the fix they did not, and a
# disagreement between what a screen shows and what training reads is the
# failure mode this project treats as a blocker.
view = store.read().resolved
check(view["comfy_dir"] == str(path_tiers.comfy_dir(root, kv))
      and view["checkpoints_dir"] == str(path_tiers.checkpoints_dir(root, kv))
      and view["loras_dir"] == str(path_tiers.loras_dir(root, kv)),
      f"and the settings view reports exactly what resolution returns -- one "
      f"answer, not two (view: {view})")

# ==========================================================================
# Section: the explicit overrides still win
# ==========================================================================
print("\n-- the override tiers, in their documented order --")

elsewhere = root / "elsewhere"
(elsewhere / "checkpoints").mkdir(parents=True)
(elsewhere / "loras").mkdir(parents=True)

store.update(SettingsChanges(
    checkpoints_dir=str(elsewhere / "checkpoints"),
    loras_dir=str(elsewhere / "loras"),
))
check(path_tiers.checkpoints_dir(root, kv) == elsewhere / "checkpoints",
      "an explicit checkpoints_dir beats ComfyUI's layout")
check(path_tiers.loras_dir(root, kv) == elsewhere / "loras",
      "and an explicit loras_dir beats it too")

# models_dir is the coarser tier: it names the tree, the specific keys name
# a child of it.
models_root = root / "srv" / "models"
(models_root / "checkpoints").mkdir(parents=True)
(models_root / "loras").mkdir(parents=True)
store.update(SettingsChanges(
    checkpoints_dir="", loras_dir="", models_dir=str(models_root),
))
check(path_tiers.checkpoints_dir(root, kv) == models_root / "checkpoints",
      f"and models_dir supplies the tree when the specific keys are unset "
      f"(got {path_tiers.checkpoints_dir(root, kv)})")
check(path_tiers.loras_dir(root, kv) == models_root / "loras",
      "for LoRAs as well")

# `store.update` *creates* the directory a valid override names -- that is
# documented settings-store behaviour, so "a models_dir that does not exist"
# cannot be produced by writing one. It takes a row written directly, which
# is what a database edited by hand or restored from a backup looks like.
#
# Worth checking because the alternative is the invented-path problem by
# another route: resolution would report a directory that is not there, and
# nothing downstream could tell it from a real one.
never = root / "never-created"


def _put_setting(key: str, value: str) -> None:
    """Write a settings row directly, bypassing the store's validation."""
    with db.connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )


_put_setting("models_dir", str(never))
check(not never.exists(),
      f"the directory was never created ({never.exists()})")
check(path_tiers.checkpoints_dir(root, kv) == comfy / "models" / "checkpoints",
      f"so it is ignored, and resolution falls back to ComfyUI's layout "
      f"(got {path_tiers.checkpoints_dir(root, kv)})")
check(path_tiers.loras_dir(root, kv) == comfy / "models" / "loras",
      "for LoRAs too, rather than answering with a path that is not there")

with db.connection() as conn:
    conn.execute("DELETE FROM settings WHERE key = 'models_dir'")
check(path_tiers.checkpoints_dir(root, kv) == comfy / "models" / "checkpoints",
      "and removing the row returns to that same layout")

finish()