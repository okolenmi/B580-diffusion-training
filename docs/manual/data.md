# Data and shapes

Part of the trainer manual (`../MANUAL.md` has the short version). Status: `verified` = run on the B580, `untested` = unit-tested in the sandbox only.

| Item | What it does |
|---|---|
| Auto caption | `--auto-caption` (model and options: `--param autocaption.model_id=...`, `prefix`, `suffix`, `lowercase`, `limit`, `overwrite`, `channel`) | Before training, a Hugging Face image-to-text model (BLIP base by default) writes captions into the dataset's library for images whose caption is empty, so a captionless set stops giving cross-attention one constant conditioning. Resumable (only empty captions are written), commits as it goes, frees the model afterwards. Backends are pluggable (`register_captioner`). | Time for one pass over the images, once; downloads the model on first use. | untested on a real model (logic verified with a stub) |
| `latents.base` / `--resolution` | Short side in pixels (768 = SDXL-ish low budget, 1024 full). |
| `latents.max_aspect` (2.0) | Longest side is at most this times base. |
| `buckets.multiple` | Latent padding multiple (4 latents = 32 px). |
| `buckets.max_shapes` | See VRAM/Speed. |
| `buckets.batch_area` | See VRAM. |
| `buckets.pack` | See VRAM (fewer short batches). |
| `buckets.keep_incomplete` | Train on a short last batch (a shape of its own: costs a compile) or drop it. |
| `batcher.shuffle` | Shuffles batches. Same-shape batches arrive in clumps of up to 4. |
| `data:` line (console) | Epoch size, shapes, short batches, packed and dropped samples, padding share (of what the plan really runs); check it after changing any of the above. |
