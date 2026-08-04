# `nodes/components/`

Destination for rewritten, non-legacy versions of the technical pieces
`nodes/` currently imports from `core/` or `manager/` -- dataset loading
(`manager.loader.ManagedDatasetLoader`), UNet/LoRA construction
(`core.unet_wrapper`, `core.lora`), noise-schedule math
(`core.noise_schedule`), model I/O parameterization (`core.model_io`),
and similar. See `docs/nodes_package_design.md` for the full picture of
which subpackage currently depends on what.

`diffusion.py` (`NoiseSchedule`/`Parameterization`/`ModelInputTransform`/
`DiffusionProcess`) and `device.py` (`DeviceContext`) have landed --
backlog items 1-2, equivalence-tested against `core.noise_schedule`/
`core.model_io`/`core.comfy_setup` in
`nodes/smoke_tests/smoke_test_diffusion_equivalence.py` and
`smoke_test_device_context_equivalence.py`, and wired into
`nodes/train/supervised.py`'s `_run_step` in place of the three direct
`core.*` imports it used to reach for. `core.noise_schedule`/
`core.model_io`/`core.comfy_setup` themselves are untouched (reference
material, per the project's existing rule) -- `nodes/dataset/renoise.py`
still imports them directly and is intentionally out of scope for this
slice (its `sample_timestep` usage isn't part of the `NoiseSchedule`/
`Parameterization` contract at all).

Everything else in the "destination for" list above is still to move.
See `docs/training_pipeline_design.md`'s "Prioritized backlog" (section
10) for the current ordered plan of what lands here next and why the
bigger items (a full step-pipeline refactor, cross-component offload
orchestration) are sequenced later.

When something does move here: equivalence-test it against the
`core`/`manager` code it replaces before anything switches over to it,
same discipline `nodes/optimizer/` already used (see that subpackage's
`Algorithm` classes and their smoke tests for the pattern). A rewrite
that isn't checked against the thing it replaces isn't done yet.
