"""Correctness check for ProjectLayout (docs/training_pipeline_design.md
section 1.6, nodes/components/layout.py).

The crux is check_frozen_snapshot_does_not_drift() -- the entire point of
this class is that it's a value object, immutable after construction,
unlike paths.py's module-global state. Everything else here is
equivalence against paths.py's own functions (same discipline as every
other new object in this backlog).

This test mutates paths.py's module-global override state (via
set_comfy_dir/set_checkpoints_dir/set_loras_dir) to exercise both sides
of the bridge -- it restores everything to unset in a finally block, so
it's safe to run alongside anything else in the same process.

Run this directly: `python nodes/smoke_tests/smoke_test_project_layout.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import paths
from nodes.components.layout import ProjectLayout

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def check_from_paths_module_matches_live_state():
    print("\n=== from_paths_module() matches paths.py's live state exactly ===")
    layout = ProjectLayout.from_paths_module()
    record(layout.comfy_dir == paths.get_comfy_dir(), "comfy_dir matches get_comfy_dir()")
    record(layout.checkpoints_dir == paths.get_checkpoints_dir(),
           "checkpoints_dir matches get_checkpoints_dir()")
    record(layout.loras_dir == paths.get_loras_dir(), "loras_dir matches get_loras_dir()")
    record(layout.datasets_dir == paths.get_datasets_dir(),
           "datasets_dir matches get_datasets_dir()")
    record(layout.runs_dir == paths.get_runs_dir(), "runs_dir matches get_runs_dir()")


def check_resolve_methods_match_paths_module():
    print("\n=== resolve_model_path/resolve_safe_model_path match paths.py's own ===")
    layout = ProjectLayout.from_paths_module()

    record(layout.resolve_model_path("foo.safetensors", "checkpoint")
           == paths.resolve_model_path("foo.safetensors", "checkpoint"),
           "resolve_model_path matches paths.resolve_model_path")
    record(layout.resolve_safe_model_path("sub/foo.safetensors", "lora")
           == paths.resolve_safe_model_path("sub/foo.safetensors", "lora"),
           "resolve_safe_model_path matches paths.resolve_safe_model_path")
    record(layout.resolve_dataset_path("myset")
           == paths.resolve_dataset_path("myset"),
           "resolve_dataset_path matches paths.resolve_dataset_path")
    record(layout.resolve_safe_dataset_path("myset")
           == paths.resolve_safe_dataset_path("myset"),
           "resolve_safe_dataset_path matches paths.resolve_safe_dataset_path")

    for bad in ("../escape", "/absolute", ""):
        layout_raised = paths_raised = False
        try:
            layout.resolve_safe_model_path(bad, "checkpoint")
        except ValueError:
            layout_raised = True
        try:
            paths.resolve_safe_model_path(bad, "checkpoint")
        except ValueError:
            paths_raised = True
        record(layout_raised and paths_raised,
               f"both reject {bad!r} the same way",
               detail=f"layout_raised={layout_raised}, paths_raised={paths_raised}")

    try:
        layout.resolve_model_path("x", "bogus_kind")
        ok = False
    except ValueError:
        ok = True
    record(ok, "resolve_model_path rejects an invalid kind")


def check_frozen_snapshot_does_not_drift():
    """THE CRUX: a constructed ProjectLayout must keep reporting the
    directories it was built with, even after paths.py's global state
    changes underneath it -- that's the entire reason this class exists."""
    print("\n=== THE FIX: a constructed ProjectLayout doesn't drift when paths.py's globals change ===")
    with tempfile.TemporaryDirectory() as old_ckpt, tempfile.TemporaryDirectory() as new_ckpt:
        paths.set_checkpoints_dir(old_ckpt)
        layout = ProjectLayout.from_paths_module()
        record(str(layout.checkpoints_dir) == str(Path(old_ckpt).resolve()),
               "snapshot captured the directory active at construction time")

        paths.set_checkpoints_dir(new_ckpt)
        record(str(layout.checkpoints_dir) == str(Path(old_ckpt).resolve()),
               "existing ProjectLayout instance is UNCHANGED after the global state moved on",
               detail=f"got {layout.checkpoints_dir}")
        record(paths.get_checkpoints_dir() == Path(new_ckpt).resolve(),
               "paths.py's own module state, meanwhile, DID move to the new directory "
               "-- both mechanisms are individually correct, just at different times")

        fresh_layout = ProjectLayout.from_paths_module()
        record(str(fresh_layout.checkpoints_dir) == str(Path(new_ckpt).resolve()),
               "a FRESH from_paths_module() call picks up the new directory")


def check_is_frozen():
    print("\n=== ProjectLayout is actually immutable (frozen dataclass) ===")
    layout = ProjectLayout.from_paths_module()
    try:
        layout.checkpoints_dir = Path("/somewhere/else")
        ok = False
    except Exception:
        ok = True
    record(ok, "assigning to a field raises (FrozenInstanceError)")


def check_construction_tolerates_unresolvable_comfy_dir():
    """The actual bug this test caught during development: a caller that
    only ever needed loras_dir/checkpoints_dir (which have their own
    internal `except RuntimeError` fallback in paths.py) must not be
    blocked by comfy_dir being unresolvable -- get_comfy_dir() raises in
    exactly this environment (no COMFY_DIR set, no ComfyUI found nearby),
    which is also LoRACheckpointLoaderNode's actual real test environment
    in smoke_test_lora_checkpoint_loader.py, not a contrived case."""
    print("\n=== from_paths_module() tolerates an unresolvable comfy_dir ===")
    paths.set_comfy_dir(None)  # simulate "genuinely not configured"
    import os
    old_env = os.environ.pop("COMFY_DIR", None)
    try:
        try:
            paths.get_comfy_dir()
            print("  (skipped -- this sandbox unexpectedly resolves a comfy_dir; "
                  "the case this check targets isn't reachable here)")
            return
        except RuntimeError:
            pass  # good, this is the case we want to exercise

        try:
            layout = ProjectLayout.from_paths_module()
            ok = True
        except RuntimeError as e:
            ok = False
            record(ok, "construction succeeds despite comfy_dir being unresolvable",
                   detail=repr(e))
            return
        record(ok, "construction succeeds despite comfy_dir being unresolvable")
        record(layout.loras_dir == paths.get_loras_dir(),
               "loras_dir still resolves correctly (via its own fallback)")
    finally:
        if old_env is not None:
            os.environ["COMFY_DIR"] = old_env


def main():
    check_construction_tolerates_unresolvable_comfy_dir()

    original_comfy = tempfile.mkdtemp()  # get_comfy_dir()'s override requires .exists()
    paths.set_comfy_dir(original_comfy)
    try:
        check_from_paths_module_matches_live_state()
        check_resolve_methods_match_paths_module()
        check_frozen_snapshot_does_not_drift()
        check_is_frozen()
    finally:
        paths.set_comfy_dir(None)
        paths.set_checkpoints_dir(None)
        paths.set_loras_dir(None)

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
