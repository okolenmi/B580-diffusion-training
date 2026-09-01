"""Real torch, real safetensors I/O against a temp directory
(paths.set_checkpoints_dir) -- verifies Phase 1 of
docs/resources_controller_redesign_plan.md: SafetensorsCheckpointNode
no longer eagerly loads a checkpoint's tensor data, ModelWeights
(nodes/model/handle.py) lazily materializes on first real access and
caches after that, and nodes/model/resource_inspection.py's header-only
dtype inspection is both cheap (never touches the lazy full-load path)
and correct (matches what a real load actually produces).

The core property under test throughout: nothing about this should be
observable as a *behavior* change to an existing consumer -- only as a
*timing* change (when the real load happens) and a new capability
(inspect before committing to a load). Every check below that asserts
"identical to the old eager path" is checking exactly that.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from safetensors.torch import save_file

import paths
from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
from nodes.model.handle import ModelWeights
from nodes.model.resource_inspection import (
    ComponentDtype, LoRAInspection, classify_key, inspect_checkpoint_dtypes, inspect_lora,
)


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _make_sdxl_like_checkpoint(path, unet_dtype=torch.bfloat16, clip_dtype=torch.float16,
                                vae_dtype=torch.float16, mixed_vae=False):
    """A small fixture with real keys under each of the three real SDXL
    prefixes this project's own checkpoint format uses -- not
    arbitrary/simplified key names, the actual ones
    resource_inspection.py's classify_key() is grounded in."""
    tensors = {
        "model.diffusion_model.input_blocks.0.weight": torch.randn(4, 4, dtype=unet_dtype),
        "model.diffusion_model.out.weight": torch.randn(4, 4, dtype=unet_dtype),
        "conditioner.embedders.0.transformer.weight": torch.randn(4, 4, dtype=clip_dtype),
        "conditioner.embedders.1.transformer.weight": torch.randn(4, 4, dtype=clip_dtype),
        "first_stage_model.encoder.weight": torch.randn(4, 4, dtype=vae_dtype),
        "first_stage_model.decoder.weight": torch.randn(
            4, 4, dtype=(torch.float32 if mixed_vae else vae_dtype)),
    }
    save_file(tensors, str(path))
    return tensors


def check_classify_key():
    print("[classify_key() matches this project's real SDXL checkpoint prefixes]")
    check(classify_key("model.diffusion_model.input_blocks.0.weight") == "unet", "unet prefix")
    check(classify_key("first_stage_model.encoder.weight") == "vae", "vae prefix")
    check(classify_key("conditioner.embedders.0.transformer.weight") == "clip",
          "everything else (SDXL's two real text-encoder prefixes) is masked as one CLIP bucket")
    print("    PASS")


def check_inspect_checkpoint_dtypes_uniform_case(tmpdir):
    print("[inspect_checkpoint_dtypes(): correct dtype per component, uniform case]")
    path = tmpdir / "uniform.safetensors"
    _make_sdxl_like_checkpoint(path, unet_dtype=torch.bfloat16, clip_dtype=torch.float16,
                                vae_dtype=torch.float16)
    result = inspect_checkpoint_dtypes(path)
    check(result["unet"] == ComponentDtype(dtype=torch.bfloat16, key_count=2), result["unet"])
    check(result["clip"] == ComponentDtype(dtype=torch.float16, key_count=2), result["clip"])
    check(result["vae"] == ComponentDtype(dtype=torch.float16, key_count=2), result["vae"])
    print("    PASS")


def check_inspect_checkpoint_dtypes_mixed_and_absent(tmpdir):
    print("[inspect_checkpoint_dtypes(): mixed dtype within a component is reported as "
          "None with a real key_count, not silently resolved to a majority; a component "
          "with zero keys is also None but with key_count=0 -- distinguishable from mixed]")
    path = tmpdir / "mixed.safetensors"
    _make_sdxl_like_checkpoint(path, mixed_vae=True)
    result = inspect_checkpoint_dtypes(path)
    check(result["vae"].dtype is None and result["vae"].key_count == 2,
          f"expected mixed vae (dtype=None, key_count=2), got {result['vae']}")

    unet_only_path = tmpdir / "unet_only.safetensors"
    save_file({"model.diffusion_model.out.weight": torch.randn(2, 2, dtype=torch.bfloat16)},
              str(unet_only_path))
    result2 = inspect_checkpoint_dtypes(unet_only_path)
    check(result2["vae"] == ComponentDtype(dtype=None, key_count=0),
          f"absent component should be key_count=0, got {result2['vae']}")
    check(result2["clip"] == ComponentDtype(dtype=None, key_count=0),
          f"absent component should be key_count=0, got {result2['clip']}")
    check(result2["unet"] == ComponentDtype(dtype=torch.bfloat16, key_count=1), result2["unet"])
    print("    PASS")


def check_safetensors_checkpoint_node_no_longer_eagerly_loads(tmpdir):
    print("[SafetensorsCheckpointNode.build() no longer loads tensor data -- only a "
          "header-only existence/format check]")
    path = tmpdir / "lazy_check.safetensors"
    real_tensors = _make_sdxl_like_checkpoint(path)

    import safetensors.torch as st_torch
    load_calls = []
    original_load_file = st_torch.load_file
    st_torch.load_file = lambda *a, **kw: (load_calls.append(1), original_load_file(*a, **kw))[1]
    try:
        node = SafetensorsCheckpointNode()
        result = node.build(path="lazy_check.safetensors")
        check(len(load_calls) == 0,
              f"build() itself should never call load_file() -- got {len(load_calls)} call(s)")
        weights = result["weights"]
        check(isinstance(weights, ModelWeights), type(weights))

        # First real access triggers exactly one load...
        _ = weights.unet_sd
        check(len(load_calls) == 1, f"expected exactly 1 load_file() call, got {len(load_calls)}")
        # ...and a second access (either property) doesn't load again.
        _ = weights.unet_sd
        _ = weights.non_unet_sd
        check(len(load_calls) == 1,
              f"repeated access should be cached, not reload -- got {len(load_calls)} total calls")
    finally:
        st_torch.load_file = original_load_file

    # And the eventually-materialized data is genuinely identical to what
    # the old eager path would have produced -- same keys, same values,
    # same dtypes, not just "didn't crash."
    for key, tensor in real_tensors.items():
        if key.startswith("model.diffusion_model."):
            check(key in weights.unet_sd, f"missing from unet_sd: {key}")
            torch.testing.assert_close(weights.unet_sd[key], tensor)
        else:
            check(key in weights.non_unet_sd, f"missing from non_unet_sd: {key}")
            torch.testing.assert_close(weights.non_unet_sd[key], tensor)
    check(len(weights.unet_sd) == 2 and len(weights.non_unet_sd) == 4,
          "non_unet_sd must still be the combined clip+vae bucket, unchanged -- "
          "Phase 1 didn't touch that split's own semantics, only when it runs")
    print("    PASS")


def check_model_weights_inspect_dtypes_matches_real_load(tmpdir):
    print("[ModelWeights.inspect_dtypes() matches the dtypes a real load actually "
          "produces, and never triggers that real load itself]")
    path = tmpdir / "cross_check.safetensors"
    _make_sdxl_like_checkpoint(path, unet_dtype=torch.bfloat16, clip_dtype=torch.float16,
                                vae_dtype=torch.float16)
    weights = ModelWeights(path)
    inspected = weights.inspect_dtypes()
    check(weights._loaded is False,
          "inspect_dtypes() must not trigger the lazy full-load cache")

    real_unet_dtype = next(iter(weights.unet_sd.values())).dtype
    real_non_unet_dtypes = {t.dtype for t in weights.non_unet_sd.values()}
    check(inspected["unet"].dtype == real_unet_dtype,
          f"{inspected['unet'].dtype} vs real {real_unet_dtype}")
    check(inspected["clip"].dtype in real_non_unet_dtypes, inspected["clip"].dtype)
    check(inspected["vae"].dtype in real_non_unet_dtypes, inspected["vae"].dtype)
    print("    PASS")


def check_from_state_dicts_still_works_for_the_eager_case():
    print("[ModelWeights.from_state_dicts(): the eager, no-file construction path "
          "this class unconditionally used before Phase 1 -- still real, still works]")
    unet_sd = {"model.diffusion_model.x": torch.zeros(2)}
    non_unet_sd = {"conditioner.embedders.0.x": torch.zeros(2)}
    weights = ModelWeights.from_state_dicts(unet_sd, non_unet_sd)
    check(weights.unet_sd is unet_sd, "should be the exact same dict, not a copy")
    check(weights.non_unet_sd is non_unet_sd, "should be the exact same dict, not a copy")

    try:
        weights.inspect_dtypes()
        raise AssertionError("expected a clear RuntimeError -- no file to inspect")
    except RuntimeError as e:
        check("from_state_dicts" in str(e), str(e))
        print(f"    PASS (clear error, not a crash): {e}")


def check_bad_checkpoint_path_fails_clearly_at_build_time(tmpdir):
    print("[a path that isn't a real safetensors file fails at build() time, with a "
          "clear message -- not silently deferred to whenever something first "
          "touches .unet_sd]")
    bad_path = tmpdir / "not_a_checkpoint.safetensors"
    bad_path.write_bytes(b"this is not a safetensors file")
    node = SafetensorsCheckpointNode()
    try:
        node.build(path="not_a_checkpoint.safetensors")
        raise AssertionError("expected a ValueError at build() time")
    except ValueError as e:
        check("doesn't look like a valid safetensors file" in str(e), str(e))
        print(f"    PASS: {e}")


def check_inspect_lora_uniform_case(tmpdir):
    print("[inspect_lora(): correct dtype and rank for a real, uniform saved LoRA]")
    path = tmpdir / "test_lora.safetensors"
    tensors = {
        "lora_unet_input_blocks_0.lora_down.weight": torch.randn(16, 8, dtype=torch.float32),
        "lora_unet_input_blocks_0.lora_up.weight": torch.randn(4, 16, dtype=torch.float32),
        "lora_unet_input_blocks_0.alpha": torch.tensor([16.0]),
        "lora_unet_out.lora_down.weight": torch.randn(16, 4, dtype=torch.float32),
        "lora_unet_out.lora_up.weight": torch.randn(2, 16, dtype=torch.float32),
        "lora_unet_out.alpha": torch.tensor([16.0]),
    }
    save_file(tensors, str(path))
    result = inspect_lora(path)
    check(result == LoRAInspection(dtype=torch.float32, rank=16, key_count=2), result)
    print("    PASS")


def check_inspect_lora_mixed_and_absent(tmpdir):
    print("[inspect_lora(): mixed rank/dtype reported as None with a real key_count, "
          "not silently resolved to a majority; a non-LoRA file is key_count=0]")
    mixed_path = tmpdir / "mixed_rank.safetensors"
    save_file({
        "lora_unet_a.lora_down.weight": torch.randn(8, 4, dtype=torch.float32),
        "lora_unet_a.lora_up.weight": torch.randn(2, 8, dtype=torch.float32),
        "lora_unet_b.lora_down.weight": torch.randn(16, 4, dtype=torch.float16),
        "lora_unet_b.lora_up.weight": torch.randn(2, 16, dtype=torch.float16),
    }, str(mixed_path))
    result = inspect_lora(mixed_path)
    check(result.dtype is None and result.rank is None and result.key_count == 2, result)

    not_a_lora_path = tmpdir / "not_a_lora.safetensors"
    save_file({"model.diffusion_model.x.weight": torch.randn(2, 2)}, str(not_a_lora_path))
    result2 = inspect_lora(not_a_lora_path)
    check(result2 == LoRAInspection(dtype=None, rank=None, key_count=0), result2)
    print("    PASS")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        paths.set_checkpoints_dir(tmpdir)
        try:
            check_classify_key()
            check_inspect_checkpoint_dtypes_uniform_case(tmpdir)
            check_inspect_checkpoint_dtypes_mixed_and_absent(tmpdir)
            check_safetensors_checkpoint_node_no_longer_eagerly_loads(tmpdir)
            check_model_weights_inspect_dtypes_matches_real_load(tmpdir)
            check_from_state_dicts_still_works_for_the_eager_case()
            check_bad_checkpoint_path_fails_clearly_at_build_time(tmpdir)
            check_inspect_lora_uniform_case(tmpdir)
            check_inspect_lora_mixed_and_absent(tmpdir)
        finally:
            paths.set_checkpoints_dir(None)

    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
