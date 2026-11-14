# Trainer manual

Short version: how to set things, what to run for the usual goals, and where the details are. Everything below the
recipes is a lookup, read only the page you need.

## How to set things

Everything goes through `python -m next.run sdxl-lora ...`.

- **Common options** have their own flag: `--batch 2 --resolution 768 --steps 200 --rank 16 --max-shapes 16 --graph ...` (`--help` lists them).
- **Anything else is `--set NAME=VALUE`** (repeat it as often as needed). The name decides what it is:
  a graph setting (`memory.vram_max_mb`, `precision.denoiser`, `seed.base`, `runtime.profile`, ...) or a node parameter of the starter graph
  written `node.port` (`trainer.tune_checkpointing`, `adapter.attention_fraction`, `blend.strength_max`, ...). `--param` still works and means node parameters only.
- **Node names** of the starter graph: `model images latents captions tag_swap buckets noiser batcher blend conditioner adapter optimizer
  schedule weighting trainer trigger saver` (and `balance`, `autocaption`, `epoch_trigger` when their flags are on).
  `--dump graph.json` writes the graph without running it, to see every node and port.
- `--memory-log run.csv` samples memory four times a second against what the run is doing; `--verbose` prints every event.

## What to run for what

| Goal | Run |
|---|---|
| Fastest steps | `--graph` (replays the step as a device graph; each shape is timed against eager and dropped when it does not win, which on this card is the case for bf16; raise `--set trainer.graph_shapes=12` while memory allows), `--set trainer.tune_checkpointing=true --set trainer.checkpoint_by=measured` (recomputes only what must be), `--persist-conditioning` |
| Least VRAM | checkpointing on (default), `--set optimizer.state_precision=int8_blockwise` (small), lower `--rank`, `--batch 1 --accumulation N`; expandable allocator segments are on by default (measured 8.3 -> 7.1 GB) |
| Bigger batches on 12 GB | `--batch 8 --batch-area 1.5` and an explicit `--set memory.vram_max_mb=11000`: batches that would not fit are split into equal chunks by the planner; packing (`--pack`, default 0.3) turns short batches into full ones |
| A run you leave alone | `--epochs 3 --save-every-epoch 1` (no step count needed), `--persist-conditioning`, `--set memory.vram_max_mb=...` so another program does not push it out of memory |
| Weak noise ranges | `--time-balance normalize` (or `dro`), `--balance-sampling 1`; see them first with `--set trainer.probe_every=50` |
| Better captions | `--auto-caption` (fills empty captions), tag nodes (shuffle, swap, dropout), `--set blend.strength_max=0.5` (blend toward the empty prompt) |
| Understand where memory goes | `--memory-log run.csv`; the console ledger rows; the pages below |

Combinations that do not work together are refused with a message (graph with `--fused` or dropout; fused with int8 state). `group_size=1`
with int8 state was 2x slower, leave it.

## Details by topic

| Page | Contents |
|---|---|
| [manual/vram.md](manual/vram.md) | every memory lever: batch area, packing, shapes, checkpointing, budget and planner, optimizer, precision, allocator |
| [manual/performance.md](manual/performance.md) | the performance lab: profile a run, rate every piece against the card's measured ceilings, matrix of runs, micro-benchmarks, the one-command hardware bundle |
| [manual/acceptance.md](manual/acceptance.md) | automatic verdicts for the hardware runs, per-region quantization probe, padding sheet |
| [manual/lab.md](manual/lab.md) | the lab: train variants of a LoRA, render the same images with each (settings stored in every PNG), measure fine detail and loss by timestep, one report |
| [manual/speed.md](manual/speed.md) | graph replay, measured checkpoint placement, caches, prefetch, timing |
| [manual/quality.md](manual/quality.md) | learning rate, loss weighting, time balance, padding and masks, seeds, run length, resume, merge adapter, blend |
| [manual/data.md](manual/data.md) | dataset and bucket parameters, auto caption, the `data:` line |
| [manual/diagnostics.md](manual/diagnostics.md) | memory log, ledger, events, probe, gradient alignment, warnings |
| [manual/persistence-and-unbuilt.md](manual/persistence-and-unbuilt.md) | what is kept on disk; settings that exist but do nothing yet |

Each lever has a status: `verified` = run on the B580 and behaved as described; `untested` = unit-tested in the sandbox only (the
matching entry in `TESTING.md` says how to check it).
