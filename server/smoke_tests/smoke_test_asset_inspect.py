"""Real torch, real safetensors I/O, real sandboxing -- targets
server/asset_paths.py's inspect() directly (the real logic
/nodegraph/assets/{kind}/inspect thinly wraps, same "test the function
the route delegates to, not the HTTP layer" convention as
smoke_test_execution_registry.py), for Phase 2 of
docs/resources_controller_redesign_plan.md.

Three things checked, deliberately kept separate: (1) the real answer
is right for both supported kinds (checkpoint, lora), checked
field-by-field against an explicit allowlist -- not just "the response
looks about right"; (2) the adversarial cases this endpoint has to get
right *because* it's reachable from the graph editor over the network,
not despite it -- path traversal, a nonexistent file, an unsupported
kind, a file that isn't really a safetensors file; (3) the response is
provably narrow -- no key beyond the documented contract, which is
what "protected" means for this endpoint in practice (the explicit
requirement behind it: the server must never hand back "out of scope"
information).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from safetensors.torch import save_file

import paths
from server import asset_paths


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _make_sdxl_like_checkpoint(path):
    tensors = {
        "model.diffusion_model.input_blocks.0.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "model.diffusion_model.out.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "conditioner.embedders.0.transformer.weight": torch.randn(4, 4, dtype=torch.float16),
        "first_stage_model.encoder.weight": torch.randn(4, 4, dtype=torch.float16),
    }
    save_file(tensors, str(path))


def check_real_answer_matches_an_explicit_allowlist(tmpdir):
    print("[inspect(): the real answer, checked field-by-field against an explicit "
          "allowlist -- not just 'looks about right']")
    _make_sdxl_like_checkpoint(tmpdir / "real.safetensors")
    result = asset_paths.inspect("checkpoint", "real.safetensors")

    check(result == {
        "kind": "checkpoint",
        "path": "real.safetensors",
        "components": {
            "unet": {"dtype": "bfloat16", "key_count": 2},
            "clip": {"dtype": "float16", "key_count": 1},
            "vae": {"dtype": "float16", "key_count": 1},
        },
    }, result)
    print("    PASS")


def check_response_is_provably_narrow(tmpdir):
    print("[the response never carries more than its documented contract -- "
          "top-level keys and each component's own keys, both checked]")
    _make_sdxl_like_checkpoint(tmpdir / "narrow.safetensors")
    result = asset_paths.inspect("checkpoint", "narrow.safetensors")

    check(set(result.keys()) == {"kind", "path", "components"},
          f"unexpected top-level keys: {set(result.keys())}")
    for name, component in result["components"].items():
        check(set(component.keys()) == {"dtype", "key_count"},
              f"component {name!r} carries unexpected keys: {set(component.keys())}")
    print("    PASS")


def check_path_traversal_rejected_the_same_way_as_everywhere_else(tmpdir):
    print("[path traversal / absolute / empty path -- rejected the same way "
          "resolve_safe_model_path already rejects it elsewhere in this project]")
    _make_sdxl_like_checkpoint(tmpdir / "real.safetensors")
    for bad in ("../escape.safetensors", "/etc/passwd", ""):
        try:
            asset_paths.inspect("checkpoint", bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError as e:
            print(f"    PASS ({bad!r}): {e}")


def check_nonexistent_file_fails_clearly():
    print("[a path that resolves safely but doesn't exist -- clear ValueError, not a crash]")
    try:
        asset_paths.inspect("checkpoint", "does_not_exist.safetensors")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        check("No such file" in str(e), str(e))
        print(f"    PASS: {e}")


def check_directory_is_rejected(tmpdir):
    print("[a directory, not a file -- clear ValueError, not an attempt to inspect it]")
    (tmpdir / "a_directory").mkdir()
    try:
        asset_paths.inspect("checkpoint", "a_directory")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        check("Not a file" in str(e), str(e))
        print(f"    PASS: {e}")


def check_corrupt_file_fails_clearly(tmpdir):
    print("[a file that exists but isn't a real safetensors file -- clear ValueError]")
    bad = tmpdir / "corrupt.safetensors"
    bad.write_bytes(b"not a safetensors file")
    try:
        asset_paths.inspect("checkpoint", "corrupt.safetensors")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        check("doesn't look like a valid safetensors file" in str(e), str(e))
        print(f"    PASS: {e}")


def check_dataset_kind_is_rejected():
    print("[kind='dataset' -- explicitly unsupported (only checkpoint/lora "
          "inspection exist), clear message]")
    try:
        asset_paths.inspect("dataset", "whatever.safetensors")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        check("checkpoint" in str(e) and "lora" in str(e), str(e))
        print(f"    PASS: {e}")


def check_lora_inspection_real_answer(tmpdir):
    print("[kind='lora': the real answer -- dtype and rank, checked against an "
          "explicit allowlist]")
    path = tmpdir / "test.safetensors"
    tensors = {
        "lora_unet_input_blocks_0.lora_down.weight": torch.randn(16, 8, dtype=torch.float32),
        "lora_unet_input_blocks_0.lora_up.weight": torch.randn(4, 16, dtype=torch.float32),
        "lora_unet_input_blocks_0.alpha": torch.tensor([16.0]),
    }
    save_file(tensors, str(path))
    result = asset_paths.inspect("lora", "test.safetensors")
    check(result == {"kind": "lora", "path": "test.safetensors", "dtype": "float32",
                      "rank": 16, "key_count": 1}, result)
    print("    PASS")


def check_lora_inspection_response_is_narrow(tmpdir):
    print("[kind='lora' response never carries more than its documented contract]")
    path = tmpdir / "test2.safetensors"
    save_file({"lora_unet_x.lora_down.weight": torch.randn(4, 4)}, str(path))
    result = asset_paths.inspect("lora", "test2.safetensors")
    check(set(result.keys()) == {"kind", "path", "dtype", "rank", "key_count"},
          f"unexpected keys: {set(result.keys())}")
    print("    PASS")


def check_lora_path_traversal_rejected(loras_dir):
    print("[kind='lora' path traversal -- rejected the same way as kind='checkpoint']")
    for bad in ("../escape.safetensors", "/etc/passwd", ""):
        try:
            asset_paths.inspect("lora", bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError as e:
            print(f"    PASS ({bad!r}): {e}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        loras_dir = tmpdir / "loras"
        loras_dir.mkdir()
        paths.set_checkpoints_dir(tmpdir)
        paths.set_loras_dir(loras_dir)
        try:
            check_real_answer_matches_an_explicit_allowlist(tmpdir)
            check_response_is_provably_narrow(tmpdir)
            check_path_traversal_rejected_the_same_way_as_everywhere_else(tmpdir)
            check_nonexistent_file_fails_clearly()
            check_directory_is_rejected(tmpdir)
            check_corrupt_file_fails_clearly(tmpdir)
            check_dataset_kind_is_rejected()
            check_lora_inspection_real_answer(loras_dir)
            check_lora_inspection_response_is_narrow(loras_dir)
            check_lora_path_traversal_rejected(loras_dir)
        finally:
            paths.set_checkpoints_dir(None)
            paths.set_loras_dir(None)

    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
