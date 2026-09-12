*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

# Phase 1 & 2 -- lazy resource references, and the server query endpoint

## Phase 1 -- Lazy resource references + header-only inspection

**Status: done.**

- `nodes/model/resource_inspection.py` (new): `classify_key()` (unet/
  vae/clip by real SDXL prefix -- `model.diffusion_model.`,
  `first_stage_model.`, everything else masked as one CLIP bucket) and
  `inspect_checkpoint_dtypes()` (header-only, via `safe_open()`,
  confirmed directly against a real file to touch no tensor data).
  Reports `ComponentDtype(dtype, key_count)` per component -- `dtype`
  is `None` for either "absent" (`key_count == 0`) or "genuinely mixed"
  (`key_count > 0`, disagreeing dtypes), kept distinguishable rather
  than collapsed into one sentinel.
- `ModelWeights` (`nodes/model/handle.py`): now constructed from a path,
  not materialized dicts. `unet_sd`/`non_unet_sd` are lazy, cached
  `@property`s -- real load only happens on first access, exactly once.
  `from_state_dicts()` classmethod keeps the old eager, no-file
  construction available (test fixtures, mainly). `inspect_dtypes()`
  delegates to `resource_inspection.py`, raises a clear error if called
  on a `from_state_dicts()` instance (nothing to peek at -- read dtype
  off the tensors directly instead).
- `SafetensorsCheckpointNode.build()` no longer calls `load_file()` --
  does a cheap header-only existence/format check (fails loudly at
  build() time on a bad file, not later when something first touches
  `.unet_sd`) and hands back a lazy `ModelWeights`.
- Zero call-site changes needed in `ComfyUNetLoRANode` or the text
  encoder node -- both still do plain `weights.unet_sd`/
  `weights.non_unet_sd` attribute access, unaware anything changed.

**Verified**, `nodes/smoke_tests/smoke_test_resource_inspection.py`:
`build()` itself never calls `load_file()`; first real access loads
exactly once, a second access doesn't reload; the eventually-
materialized data is identical (keys, values, dtypes) to what the old
eager path produced, not just "didn't crash"; `inspect_dtypes()`
matches a real load's actual dtypes and never triggers the lazy
full-load cache; the mixed/absent distinction; a bad checkpoint path
fails clearly at `build()` time. All 52 `nodes/smoke_tests` files pass
(51 before this phase + this one).

## Phase 2 -- Server query endpoint

**Status: done.**

- `GET /nodegraph/assets/{kind}/inspect?path=...` (`server/routes_nodegraph.py`),
  a thin wrapper -- same convention as `browse_assets`/`list_assets` --
  around the real logic in `server/asset_paths.py`'s new `inspect()`,
  built directly on the existing `_safe_resolve()` (same sandboxing
  `browse()`/`upload()` already use).
- Response contract, deliberately narrow: `{kind, path, components:
  {unet|clip|vae: {dtype, key_count}}}` -- nothing else. `dtype` is a
  plain string (`nodes/model/resource_inspection.py`'s new
  `dtype_to_str()`, shared rather than reimplemented -- also what a
  future validator's human-readable text line would use), not a raw
  `torch.dtype`.
- Only `kind="checkpoint"` is supported -- LoRA-file inspection needs
  its own function (different key format, different meaningful fields
  like rank) and is explicitly left as real, separate follow-up rather
  than silently folded in here.

**Verified**, `server/smoke_tests/smoke_test_asset_inspect.py` (targets
`asset_paths.inspect()` directly -- same "test the function the route
delegates to" convention as `smoke_test_execution_registry.py`, not the
HTTP layer): the real answer checked field-by-field against an explicit
allowlist; the response is provably narrow (no key beyond the
documented contract, top level and per-component); path traversal /
absolute / empty path rejected the same way `resolve_safe_model_path`
already rejects them elsewhere in this project; a nonexistent file, a
directory instead of a file, a corrupt/non-safetensors file, and an
unsupported `kind` all fail with a clear message rather than a crash or
a silent wrong answer. Router wiring confirmed (`/nodegraph/assets/
{kind}/inspect` present alongside the existing asset routes); the three
existing `server/smoke_tests/` files still pass.

**Dependency:** Phase 1 (done).
