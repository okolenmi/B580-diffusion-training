# Contract: graph settings

Python: `graph/settings.py`.

`SettingSpec` JSON (also the dialog's row description):

```jsonc
{ "key": "runtime.profile", "label": "Profile steps", "doc": "...",
  "type": "bool" | "int" | "float" | "str" | "choice",
  "default": false, "choices": null, "unit": "", "minimum": null, "maximum": null,
  "step": null, "group": "Runtime", "advanced": false, "allow_auto": false,
  "contributed_by": "core" | "<family id>" | "<node class>" }
```

Resolution: `node explicit override > graph setting > app default > spec default`.
The node override exists only where a node declares one.

Values: a value equal to the string `"auto"` is allowed only when `allow_auto`. Numbers
must satisfy `minimum`/`maximum`. Cross-key rules (core): `memory.vram_min_mb <=
memory.vram_max_mb` when both numeric.

Core keys (`graph/settings.py` holds each one's meaning): `memory.vram_max_mb`,
`memory.vram_min_mb`, `memory.strict`, `memory.ram_max_mb`, `device.default`,
`precision.denoiser`, `precision.encoders`, `precision.vae`, `precision.autocast`,
`memory.margin_mb`, `memory.allocator_mb`,
`runtime.residency.calibration_steps`, `runtime.empty_cache_every_n_steps`,
`runtime.profile`,
`conditioning.cache_mb`, `conditioning.prewarm`, `seed.base`.

Access: `ctx.settings.get("device.default")` returns the resolved value of an immutable
snapshot built at run start. A node may only read keys it lists in `READS_SETTINGS`
(enforced in `get`).

Duplicate keys across contributors are a load error.
