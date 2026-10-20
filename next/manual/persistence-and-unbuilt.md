# Persistence on disk

Part of the trainer manual (`../MANUAL.md` has the short version). Status: `verified` = run on the B580, `untested` = unit-tested in the sandbox only.

| What | Where | Control |
|---|---|---|
| Latent cache | next to the dataset | `latents.*` keys; rebuilt when they change |
| Conditioning store | next to the dataset | `persist`, `max_disk_mb` |
| Adapters | `--output`, every `--save-every` steps | `saver.name`, `format` (kohya, native) |

## Declared but not implemented

These graph settings exist in the registry and appear in the editor, but nothing reads them yet:
`runtime.graph_capture` (use the trainer's `graph` port instead), `runtime.block_swap`, `memory.ram_max_mb`. Setting them changes nothing.
Read by code: `memory.vram_max_mb`, `memory.vram_min_mb`, `memory.strict`, `device.default`, `precision.*`,
`runtime.residency.safety_margin`, `runtime.residency.calibration_steps`, `runtime.empty_cache_every_n_steps`,
`runtime.profile` (turns the trainer's `timing` on), `conditioning.cache_mb` (size of the encoded-prompt cache,
auto = 4096 MB; the Conditioner's own `cache_mb` port wins), `conditioning.prewarm` (off = encode prompts when first
needed instead of up front), `seed.base` (nodes with an empty seed derive theirs from it and their id; -1 leaves
them at 0).

Planned ports (see `rework/PORTING.md`): NF4 and int8 frozen weights (to use the owner's rewritten kernels).
