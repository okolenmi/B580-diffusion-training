# `nodes/components/`

Destination for rewritten, non-legacy versions of the technical pieces
`nodes/` currently imports from `core/` or `manager/` -- dataset loading
(`manager.loader.ManagedDatasetLoader`), UNet/LoRA construction
(`core.unet_wrapper`, `core.lora`), noise-schedule math
(`core.noise_schedule`), model I/O parameterization (`core.model_io`),
and similar. See `docs/nodes_package_design.md` for the full picture of
which subpackage currently depends on what.

Nothing has moved here yet -- this folder exists to hold that work as it
happens, not as a completed migration.

See `docs/training_pipeline_design.md`'s "Prioritized backlog" (section
10) for the current ordered plan of what lands here first (diffusion-
process math and device-backend handling, replacing
`core/noise_schedule.py`'s/`core/comfy_setup.py`'s module-level state --
see that doc's sections 1.4/1.5/9.2 for the reasoning) and why the bigger
items (a full step-pipeline refactor, cross-component offload
orchestration) are sequenced later.

When something does move here: equivalence-test it against the
`core`/`manager` code it replaces before anything switches over to it,
same discipline `nodes/optimizer/` already used (see that subpackage's
`Algorithm` classes and their smoke tests for the pattern). A rewrite
that isn't checked against the thing it replaces isn't done yet.
