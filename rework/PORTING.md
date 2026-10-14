# Porting ledger

Resume point for the rewrite (plan 90). One row per old file. Columns:
**triage** = `keep` (move with comment cleanup only), `rewrite` (read, then write fresh from the plan), `drop` (not needed).
**state** = `todo`, `lessons` (hardware lessons harvested to `lessons/`), `done`. Destinations are *planned*; confirm during triage.
Line counts and comment share are measured by script on 2026-10-13 (docstrings + `#` lines).

Totals: 133 files, 27,586 lines, 40.9% comments/docstrings (11,285 lines), 3,642 blank; roughly 12,700 lines of actual code.

| old file | lines | comment % | planned destination | triage | state |
|---|---|---|---|---|---|
| `manager/__init__.py` | 3 | 0 | next/data/ (triage) | ? | todo |
| `manager/builder.py` | 724 | 21 | next/data/ingest | ? | todo |
| `manager/caption_pool.py` | 137 | 32 | next/data/ (triage) | ? | todo |
| `manager/captioning.py` | 244 | 23 | next/data/ (triage) | ? | todo |
| `manager/dataset.py` | 265 | 18 | next/data/ (triage) | ? | todo |
| `manager/db.py` | 379 | 11 | next/data/ (triage) | ? | todo |
| `manager/loader.py` | 749 | 39 | next/data/ (batching, Noiser) | ? | todo |
| `manager/preview.py` | 57 | 14 | next/data/ (triage) | ? | todo |
| `manager/storage.py` | 172 | 14 | next/data/ (triage) | ? | todo |
| `manager/t_sampling.py` | 172 | 47 | next/data/ (triage) | ? | todo |
| `nodes/components/device.py` | 320 | 40 | next/engine/runtime/ | ? | todo |
| `nodes/components/diffusion.py` | 169 | 26 | next/families/general/schedules/ | ? | todo |
| `nodes/components/layout.py` | 118 | 55 | next/ (triage) | ? | todo |
| `nodes/components/model_io.py` | 83 | 46 | next/families/general/safetensors_io | ? | todo |
| `nodes/components/noise_schedule.py` | 95 | 38 | next/families/general/schedules/ | ? | todo |
| `nodes/components/seed.py` | 32 | 78 | next/ (triage) | ? | todo |
| `nodes/config_io.py` | 292 | 20 | next/ (triage) | ? | todo |
| `nodes/config_model.py` | 338 | 14 | next/ (triage) | ? | todo |
| `nodes/core.py` | 568 | 52 | next/graph/ | rewrite | done |
| `nodes/dataset/auto_caption.py` | 112 | 11 | next/data/ + nodes/data/ | ? | todo |
| `nodes/dataset/caption_pool.py` | 104 | 9 | next/data/ + nodes/data/ | ? | todo |
| `nodes/dataset/handle.py` | 29 | 24 | next/data/ + nodes/data/ | ? | todo |
| `nodes/dataset/managed.py` | 305 | 15 | next/data/ + nodes/data/ | ? | todo |
| `nodes/dataset/node.py` | 21 | 5 | next/data/ + nodes/data/ | ? | todo |
| `nodes/dataset/prefetch.py` | 157 | 45 | next/data/ + nodes/data/ | ? | todo |
| `nodes/dataset/timestep_modes.py` | 51 | 92 | next/data/ + nodes/data/ | ? | todo |
| `nodes/memory/control_handle.py` | 489 | 70 | next/engine/runtime/ (memory, admission) | ? | todo |
| `nodes/memory/coordinator.py` | 115 | 43 | next/engine/runtime/ (memory, admission) | ? | todo |
| `nodes/memory/device_reservations.py` | 282 | 59 | next/engine/runtime/ (memory, admission) | ? | todo |
| `nodes/memory/graph_memory.py` | 661 | 46 | next/engine/runtime/ (memory, admission) | ? | todo |
| `nodes/memory/handle.py` | 82 | 66 | next/engine/runtime/ (memory, admission) | ? | todo |
| `nodes/memory/manager.py` | 206 | 54 | next/engine/runtime/ (memory, admission) | ? | todo |
| `nodes/memory/profile.py` | 84 | 68 | next/engine/runtime/ (memory, admission) | ? | todo |
| `nodes/memory/vram_budget_controller.py` | 103 | 27 | next/engine/runtime/ (memory, admission) | ? | todo |
| `nodes/model/adapter_injection.py` | 160 | 72 | next/families/general/adapters/ | ? | todo |
| `nodes/model/adapter_strategy.py` | 206 | 43 | next/families/general/adapters/ | ? | todo |
| `nodes/model/attention.py` | 354 | 42 | next/families/general/ or sdxl/denoiser/ | ? | todo |
| `nodes/model/attention_checkpointing.py` | 227 | 78 | next/engine/runtime/ | ? | todo |
| `nodes/model/block_profiler.py` | 214 | 60 | next/engine/runtime/profiler | ? | todo |
| `nodes/model/checkpoint.py` | 312 | 50 | next/engine/runtime/ | ? | todo |
| `nodes/model/checkpoint_loader.py` | 61 | 26 | next/nodes/model/load_model.py | rewrite | done (detection only; tensor loading is phase 4) |
| `nodes/model/checkpoint_placement.py` | 92 | 54 | next/engine/runtime/ | ? | todo |
| `nodes/model/clip.py` | 554 | 40 | next/families/general/clip/ | ? | todo |
| `nodes/model/clip_encoder.py` | 331 | 50 | next/families/general/clip/ | ? | todo |
| `nodes/model/clip_state_dict.py` | 176 | 49 | next/families/general/clip/ | ? | todo |
| `nodes/model/dora_layer.py` | 319 | 46 | next/families/general/adapters/ | ? | todo |
| `nodes/model/frozen_weight_store.py` | 44 | 41 | next/families/general/ | ? | todo |
| `nodes/model/gradient_checkpointing.py` | 150 | 78 | next/engine/runtime/ | ? | todo |
| `nodes/model/handle.py` | 171 | 44 | next/families/api/ (ModelPart) | rewrite | done (parts.py) |
| `nodes/model/lora.py` | 906 | 40 | next/families/general/adapters/ | ? | todo |
| `nodes/model/lora_checkpoint_loader.py` | 189 | 35 | next/families/general/adapters/ | ? | todo |
| `nodes/model/lora_file.py` | 102 | 41 | next/families/general/adapters/ | ? | todo |
| `nodes/model/lora_gate.py` | 39 | 64 | next/families/general/adapters/ | ? | todo |
| `nodes/model/lora_injector.py` | 394 | 35 | next/families/general/adapters/ | ? | todo |
| `nodes/model/lora_merge.py` | 74 | 46 | next/families/general/adapters/ | ? | todo |
| `nodes/model/lora_phases.py` | 531 | 48 | next/families/general/adapters/ | ? | todo |
| `nodes/model/lora_saver.py` | 199 | 38 | next/families/general/adapters/ | ? | todo |
| `nodes/model/lora_scaling.py` | 60 | 55 | next/families/general/adapters/ | ? | todo |
| `nodes/model/lora_training_config.py` | 244 | 39 | next/families/general/adapters/ | ? | todo |
| `nodes/model/lora_training_resources.py` | 450 | 56 | next/families/general/adapters/ | ? | todo |
| `nodes/model/meta_init.py` | 57 | 44 | next/families/general/ | ? | todo |
| `nodes/model/nf4_lora_layer.py` | 179 | 40 | next/families/general/adapters/ or quantization/ | ? | todo |
| `nodes/model/nf4_weight_store.py` | 221 | 53 | next/families/general/adapters/ or quantization/ | ? | todo |
| `nodes/model/node.py` | 53 | 2 | next/graph/ | rewrite | done |
| `nodes/model/parameters.py` | 35 | 26 | next/ (triage) | ? | todo |
| `nodes/model/resource_inspection.py` | 149 | 38 | next/families/general/safetensors_io + sdxl/fingerprint | ? | todo |
| `nodes/model/resources_controller.py` | 477 | 46 | next/engine/runtime/ | ? | todo |
| `nodes/model/sdxl_architecture.py` | 71 | 62 | next/families/sdxl/ | ? | todo |
| `nodes/model/text_encoder.py` | 346 | 50 | next/families/general/ + sdxl/conditioning | ? | todo |
| `nodes/model/text_encoder_cache.py` | 367 | 54 | next/families/general/ + sdxl/conditioning | ? | todo |
| `nodes/model/text_encoder_prewarm.py` | 537 | 55 | next/families/general/ + sdxl/conditioning | ? | todo |
| `nodes/model/timestep_embedding.py` | 115 | 61 | next/families/sdxl/denoiser/ | ? | todo |
| `nodes/model/tokenizer.py` | 459 | 41 | next/families/general/clip/ | ? | todo |
| `nodes/model/trainer_parameters.py` | 47 | 36 | next/ (triage) | ? | todo |
| `nodes/model/unet.py` | 490 | 30 | next/families/sdxl/denoiser/ | ? | todo |
| `nodes/model/unet_wrapper.py` | 266 | 34 | next/families/sdxl/denoiser/ | ? | todo |
| `nodes/model/vae.py` | 449 | 24 | next/families/general/vae/ | ? | todo |
| `nodes/model/vae_decode.py` | 215 | 51 | next/families/general/vae/ | ? | todo |
| `nodes/monitor/handle.py` | 32 | 34 | dropped (views, plan 60) | ? | todo |
| `nodes/monitor/node.py` | 31 | 3 | dropped (views, plan 60) | ? | todo |
| `nodes/monitor/training_progress.py` | 33 | 33 | dropped (views, plan 60) | ? | todo |
| `nodes/optimizer/algorithms/adafactor.py` | 404 | 42 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/algorithms/adamw.py` | 91 | 29 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/algorithms/base.py` | 170 | 73 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/algorithms/came.py` | 321 | 39 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/composed.py` | 163 | 36 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/composed_adafactor.py` | 163 | 53 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/composed_adamw.py` | 94 | 37 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/composed_came.py` | 92 | 26 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/composed_fused.py` | 144 | 47 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/composed_fused_adafactor.py` | 107 | 47 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/composed_fused_adamw.py` | 54 | 15 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/composed_fused_came.py` | 66 | 26 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/handle.py` | 162 | 60 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/node.py` | 66 | 39 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/state_store.py` | 209 | 39 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/strategies/base.py` | 108 | 54 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/strategies/chunked.py` | 102 | 51 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/strategies/foreach.py` | 57 | 60 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/strategies/shape_grouped.py` | 49 | 47 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/strategies/shape_grouped_foreach.py` | 52 | 52 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/strategies/shape_grouping.py` | 89 | 55 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/strategies/simple.py` | 29 | 28 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/optimizer/strategy_registry.py` | 105 | 38 | next/nodes/optimizer/ (triage: port vs rewrite) | ? | todo |
| `nodes/primitive/values.py` | 72 | 8 | next/nodes/primitive/ | ? | todo |
| `nodes/quantization.py` | 57 | 56 | next/ (triage) | ? | todo |
| `nodes/resource_budget.py` | 44 | 75 | next/ (triage) | ? | todo |
| `nodes/train/bucket_balance.py` | 456 | 36 | next/data/ + nodes/data/ | ? | todo |
| `nodes/train/bucket_sizes.py` | 132 | 56 | next/data/ + nodes/data/ | ? | todo |
| `nodes/train/cache_thrash.py` | 194 | 36 | next/engine/ or nodes/train/ (triage) | ? | todo |
| `nodes/train/conditioning_blend.py` | 160 | 38 | next/nodes/conditioning/ | ? | todo |
| `nodes/train/loss.py` | 208 | 28 | next/engine/step/ + nodes | ? | todo |
| `nodes/train/managed.py` | 1714 | 40 | next/nodes/train/ (split) | ? | todo |
| `nodes/train/node.py` | 157 | 15 | next/engine/ or nodes/train/ (triage) | ? | todo |
| `nodes/train/schedule.py` | 199 | 20 | next/nodes/train/ (lr schedules) | ? | todo |
| `nodes/train/step_notify.py` | 112 | 48 | next/engine/ or nodes/train/ (triage) | ? | todo |
| `nodes/train/step_pipeline.py` | 750 | 45 | next/engine/step/ | ? | todo |
| `nodes/train/supervised.py` | 361 | 17 | dropped (one trainer) | ? | todo |
| `nodes/train/t_probe.py` | 362 | 31 | next/engine/ or nodes/train/ (triage) | ? | todo |
| `nodes/train/xpu_graph_step.py` | 336 | 32 | next/engine/runtime/capture | ? | todo |
| `nodes/ui.py` | 185 | 40 | next/graph/ (field/display descriptors) | rewrite | done |
| `nodes/xpu_env.py` | 284 | 65 | next/ (triage) | ? | todo |
