# Tests in `next/tests`

Plain scripts (`python -m next.tests.<name>`), grouped by what they protect. Each file's docstring states its purpose. Run only the group that matches what changed; nothing here needs a GPU.

| group | files | protects |
|---|---|---|
| graph | `check_graph_core` | declarations, validation, runner, events, settings |
| files and families | `check_detection`, `check_families_torch`, `check_sdxl`, `check_adapters` | recognizing model files, SDXL building blocks, adapter plans and file formats |
| data | `check_data` | epoch barrier and event order, per-VAE cache refusal, prompt determinism and resume, bucketing, dataset nodes |
| engine | `check_engine` | training steps on a tiny UNet: only adapters train, accumulation, stopping, fused equals plain, checkpointing equals plain |
| whole pipeline | `check_pipeline` | a full graph (dataset to saved adapter) through the graph runner; `pipeline_nodes.py` holds its tiny stand-ins |

Hardware checks live in `scripts/xpu_smoke.py` (real weights, real device) and are run by the owner.
