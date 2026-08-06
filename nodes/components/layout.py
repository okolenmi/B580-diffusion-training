"""ProjectLayout: resolved, immutable directory configuration for one
server/process lifetime -- replaces paths.py's module-global override
pattern (_comfy_dir_override/_checkpoints_dir_override/_loras_dir_override,
mutated via set_*(), read from anywhere via get_*()) with one constructed
value object, explicitly threaded through instead of reached into from
arbitrary call sites. See docs/training_pipeline_design.md section 1.6.

Bridging period, not a clean swap -- paths.py is genuinely used
process-wide (server/*, manager/*, core/* all depend on its module
functions today), and migrating all of that is real, separate work, out
of scope here. paths.py itself is untouched: its module functions remain
the one source of truth, and from_paths_module() below only *snapshots*
that same state into an immutable object -- it doesn't reimplement path
resolution's actual logic. resolve_model_path()/resolve_safe_model_path()
below similarly delegate to paths.py's own base-parameterized
resolve_path()/_resolve_safe_relative() (already pure functions of an
explicit `base`, not readers of global state) rather than re-deriving
that logic a second place -- particularly important for the sandboxed
variant, since getting path-traversal prevention subtly wrong by
hand-transcribing it is a real security risk, not just a style concern.
_resolve_safe_relative is private in paths.py; used directly here anyway,
deliberately, for exactly that reason -- it's the one function that
already takes an explicit base instead of reading global state, so it's
the correct thing to delegate to, underscore or not.

Wired into the four nodes/ Nodes that call paths.resolve_safe_model_path()/
resolve_safe_dataset_path() directly today (nodes/model/checkpoint_loader.py,
lora_saver.py, lora_checkpoint_loader.py, nodes/dataset/managed.py) via a
new optional project_layout Port, default None ->
ProjectLayout.from_paths_module() -- today's exact behavior, since that
reads the same paths.py global state these nodes already read directly,
just snapshotted once instead of reached into ad hoc. server/*, manager/*,
core/* are NOT touched or migrated here -- they keep using paths.py's
module functions exactly as before; both mechanisms read the same
underlying state, so they stay correct and in sync, which is what "bridging
period" means as opposed to a one-shot swap.

Deliberately not included yet: get_run_dir/get_log_path/get_progress_path/
get_dataset_db_path/get_resume_dir/list_model_files. Nothing in nodes/
calls those today (server/core do) -- adding them here now would be
speculative, not demand-driven, the same reasoning
docs/training_pipeline_design.md gives for sequencing PrefetchingBatchSource
after something actually needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectLayout:
    """Resolved, immutable set of directories for one server/process
    lifetime. Constructed once (from_paths_module(), below, or directly)
    -- not a module-global mutated by whichever setter runs first."""

    comfy_dir: Path
    checkpoints_dir: Path
    loras_dir: Path
    datasets_dir: Path
    runs_dir: Path

    @classmethod
    def from_paths_module(cls) -> "ProjectLayout":
        """Snapshots paths.py's current module-global state (whatever
        set_comfy_dir()/set_checkpoints_dir()/set_loras_dir() and the
        environment/.env currently resolve to) into one immutable object
        -- the bridge itself, see this module's docstring."""
        import paths
        try:
            comfy_dir = paths.get_comfy_dir()
        except RuntimeError:
            # get_checkpoints_dir()/get_loras_dir() already tolerate this
            # exact case (comfy_dir genuinely unconfigured/unresolvable --
            # no override, no env var, no ComfyUI found nearby) via their
            # own `except RuntimeError: return get_project_root() / ...`
            # fallback; neither actually needs comfy_dir to succeed.
            # Requiring it here, unconditionally, just to populate this
            # one field would make ProjectLayout strictly less tolerant
            # than the code it's bridging -- a real regression, caught by
            # smoke_test_lora_checkpoint_loader.py (which never configures
            # a comfy_dir and never needed one) failing against this class
            # after switching over, not by inspection.
            comfy_dir = paths.get_project_root()
        return cls(
            comfy_dir=comfy_dir,
            checkpoints_dir=paths.get_checkpoints_dir(),
            loras_dir=paths.get_loras_dir(),
            datasets_dir=paths.get_datasets_dir(),
            runs_dir=paths.get_runs_dir(),
        )

    def _base_dir(self, kind: str) -> Path:
        if kind == "checkpoint":
            return self.checkpoints_dir
        if kind == "lora":
            return self.loras_dir
        raise ValueError(f"kind must be 'checkpoint' or 'lora', got {kind!r}")

    def resolve_model_path(self, path_str: str, kind: str) -> Path:
        import paths
        return paths.resolve_path(path_str, base=self._base_dir(kind))

    def resolve_safe_model_path(self, relative_str: str, kind: str) -> Path:
        """Sandboxed variant for untrusted input (the graph editor,
        reachable over the network) -- see this module's docstring for
        why paths.py's private _resolve_safe_relative is used directly."""
        import paths
        return paths._resolve_safe_relative(relative_str, self._base_dir(kind))

    def resolve_dataset_path(self, name_or_path: str) -> Path:
        import paths
        return paths.resolve_path(name_or_path, base=self.datasets_dir)

    def resolve_safe_dataset_path(self, relative_str: str) -> Path:
        import paths
        return paths._resolve_safe_relative(relative_str, self.datasets_dir)
