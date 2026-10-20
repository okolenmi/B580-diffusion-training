# Diagnostics

Part of the trainer manual (`../MANUAL.md` has the short version). Status: `verified` = run on the B580, `untested` = unit-tested in the sandbox only.

| Tool | Gives you |
|---|---|
| Primitive-cache warning | always on | If steps that revisit an already seen shape run much slower than steps repeating one (median ratio above 1.5 after 3 revisits), one `[warning]` line says the oneDNN primitive cache is evicting shapes; raise `ONEDNN_PRIMITIVE_CACHE_CAPACITY` or lower `max_shapes`. Slow first sightings alone stay quiet (that is compilation). | None. | untested on hardware |
| Gradient alignment | `--param trainer.probe_every=50 --param trainer.probe_alignment=true` | At each probe, the gradient of the probe loss per third of the timestep range: `gc_norm/share` (which third dominates), `gc_cos_a_b` (negative = a step that helps one hurts the other), `gc_align_t_*` (cosine with the sum; <= 0: the combined update does not help that third), `gc_self_t_*` (split-half noise floor: read the other numbers relative to it). Explains `probe_rel_*`; not a trajectory. | One backward per probe point at every probe (autograd.grad, so the optimizer's gradients are untouched). Raise `probe_items`/`probe_points` to lift the noise floor. | untested on a real network (closed forms verified) |
| `--memory-log FILE.csv` | Columns: seconds, label, allocated, reserved, peak_allocated, device_used, process, active, split_free, segments, alloc_retries (MB). Labels are node/step/task names. |
| Ledger tables (console) | Additive slices of `device_used`: ours (allocated, cache, outside) and other programs. |
| Step line | `[BxHxW new]`, loss, weights, probe, timings. |
| Events | `allocator_cap`, `cache_returned`, `checkpoint_share`, `cost`, breakdown. With `--verbose`. |
| `split_free` / `segments` / `alloc_retries` | Fragmentation: free pieces stuck between live blocks; retries mean the allocator had to free and try again. |
| Task lines | Named long processes (latent build, conditioning) with progress. |
