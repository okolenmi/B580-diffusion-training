# To test on the B580

Things written in the sandbox (no XPU) that need hardware. Delete an entry once it passed; if one fails, keep it
and paste the console lines and the `--memory-log` csv. Feature background: `next/MANUAL.md`.

Common prefix used below (use your usual paths and dataset; 768, batch 2 as in earlier runs):

```
python -m next.run sdxl-lora --models <models> --datasets <datasets> --file <checkpoint> --dataset <name> \
  --output <out> --resolution 768 --batch 2 --steps 200 --memory-log run.csv
```

Run each test from the same start (fresh process). Unless said otherwise only the listed flag changes.

## 4. Tight budget, again (planner and tuners changed)
Your 10240 run held everything under 8.5 GB and could not lower the share; at batch 32 the planner predicted 13.3 GB for 22 samples that ran under 10 GB.
Cause: the cost per pixel was taken from one sample and multiplied up, so a fixed part of the step (workspaces) was counted per pixel. The probe now
runs at two batch sizes of the same shape and the planner fits fixed + growing memory.
Rerun `--set memory.vram_max_mb=10240` (batch 3, 512) with `--set trainer.tune_checkpointing=true`, and batch 32 with `--batch-area 1.5`.
- The `cost` events (two per measuring pass, `--verbose`) and `checkpoint_share`: a lower share than before may now fit at 10240.
- Batch 32: the `split` events' `predicted_mb` close to what the run really peaks at (csv `peak_allocated`), fewer or no splits, faster steps than the 8000 ms before.
Pass: predictions within ~15% of the real peak, no crash. Fail: paste the `cost`/`split` events and the csv.

Tested. Found a new issue - budget doesn't expect int8 weights (and maybe other modifications of obects in memory). With usage under 6.5GB (real process usage) it sets share at 1.0 with 10240mb limit.

Also found a problem. Previous environment variable (expandable segments) doesn't work anymore. You told me it's set on by default. The reality is memory consumption now on level as it was before that EV implementation + it doesn't work as a prefix now.
Also 

I have tried to use limit like 9000 - fail 512 batch 2:
Encoding prompts (1) ...
memory: {"event": "cache_returned", "why": "optimizer moved off the device", "returned_mb": 326, "card_freed_mb": 326, "ours_outside_mb": 0, "other_programs_mb": 0}
memory: {"event": "evict", "why": "load the text encoders", "moved": ["optimizer"]}
[   4.2s] DiffusionTrainerNode: 0.6s  allocated 6378, reserved 6392, device 7974 MB
run: failed: EventHandlerError: n_jjjjjj.prompts_changed failed on PromptsChanged: OutOfMemoryError: XPU out of memory. Tried to allocate 14.00 MiB. GPU 0 has a total capacity of 11.93 GiB of which 4.14 GiB is free. 6.24 GiB allowed; Of the allocated memory 6.23 GiB is allocated by PyTorch, and 13.99 MiB is reserved by PyTorch but unallocated. Please use `empty_cache` to release all unoccupied cached memory. 

Make A CONCRETE Memory system. I don't want to deal with such poor performance like now.


## 6. Measured checkpoint placement under a cap (retest)
Done: batch 3, 512: 930 ms against 1010 ms, peak ~10.5 GB against ~8.5. With `--set memory.vram_max_mb=10240` it was slower (1130 ms) with a `cache_returned`
event after every step, and ~950 ms with 1 GB more. Cause: after every step the manager emptied the cache because the device looked full, although the allocator
cap already makes the allocator free its own cache. That rule is off under a cap now.
Rerun `--set trainer.tune_checkpointing=true --set trainer.checkpoint_by=measured --set memory.vram_max_mb=10240`.
- No `cache_returned` after each step; steps near the ~950 ms of the roomier run.
Pass: no per-step cache events, speed near 950 ms. Fail: paste the events.

## 7. Probe: periodic lines
Your run printed one line after step 1 (1.000/1.000/1.000, as designed: the first record is the reference) and nothing after. Lines are expected every N steps.
Rerun `--set trainer.probe_every=20 --steps 100` and paste the exact command and the console; if nothing appears at steps 20, 40, ..., also run with `--verbose`
and paste a few step events (`probe_*` keys).
Note the first lines can stay 1.000 while the adapter is still near zero; after a few hundred steps they should move.

## 11. Budget cap: optimizer offloaded (fixed)
Failure: `--set memory.vram_max_mb=9216` died at step 1 with "optimizer state is offloaded; reload it before stepping".
Cause: making room for the text encoders moved the optimizer state to the host and nothing brought it back. It is now restored when the encoders leave and before every step.
Rerun `--set memory.vram_max_mb=9216` (tight on purpose), then `--set conditioning.prewarm=false`.
- `evict` events when the encoders load, `allocator_cap`, training goes on, no crash.
Pass: it trains. Fail: paste the events.

## 15. Graphs for the most used shapes
Done: replayed steps 510-550 ms against ~1000 eager, same loss. Shapes past the 8-graph cap ran eagerly, which looked like penalties (1742 ms) and
instability. Graphs now go to the shapes that have the most batches per epoch, and only those.
Rerun `--graph` (batch 2, 200 steps) with the default 8, then `--set trainer.graph_shapes=12`.
- Console: `graphs N (R replays)` rises early and then stays; eager steps come only from rare shapes; the average step time drops against your earlier run.
- Memory with 12 graphs: `device` stays under the card (the pool is shared, so it should barely move; each capture takes a few seconds).
Pass: faster average, no crash. Fail: paste the step lines around the slow steps.

## 19. Expandable segments are the default now
You measured 8.3 -> 7.1 GB with `PYTORCH_ALLOC_CONF=expandable_segments:True`; the trainer sets it itself unless you export your own value.
Run the common prefix without exporting anything.
- Peak about the same as with your manual export (~7.1 GB process). `PYTORCH_ALLOC_CONF=expandable_segments:False` restores the old behaviour for comparison.
Pass: same memory as with the manual export. Fail: tell me the number; the setting may be applied too late in the process.

## 20. One `--set`
`--set` now takes graph settings and node parameters alike: `--set memory.vram_max_mb=10240 --set trainer.graph=true --set blend.strength_max=0.5`.
- All three take effect (check with `--dump graph.json`: the trainer has `graph: true`, `settings` has the budget, the blend node has its strength).
- A typo such as `--set trainr.graph=true` stops at once and lists the node names.
Pass: both behave as described.

## 21. Experiment: kernel compile cache between runs
You saw a default run after a graph run start with almost no new-shape penalty, and the next run slow again. The driver keeps a compiled-kernel cache on disk that may fill up and evict.
Try `NEO_CACHE_MAX_SIZE=4294967296` (4 GB) before two default runs in a row, and compare the first-sighting step times (`[... new]` markers) of the second run with a run without it.
Pass: tell me whether first sightings of shapes are faster on the second run. No fix needed either way.

## 6b. Merge Adapter
Add a `Merge Adapter` node (denoiser from Load Model -> Merge Adapter -> Inject Adapter) with a kohya LoRA you have, `strength` 1 (the CLI starters have no switch for it; edit the graph).
- Task line "Merging <file>"; training starts and the step-1 loss differs from the unmerged run; a LoRA of another family or a wrong key format stops with "nothing ... matches".
- Strength 0 gives the same step-1 loss as no merge node.
Pass: all three. Fail: paste the message.

## 6c. Conditioning blend
Add `--set blend.strength_max=0.5` (min stays 0), then `--set blend.strength_min=0.5 --set blend.strength_max=0.5`.
- Trains; with `--persist-conditioning` and prewarm on, no encoder loading after the first epoch starts (the empty prompt is encoded up front).
- Fixed 0.5: loss is higher at the start than unblended and the step time is unchanged; the probe (`trainer.probe_every`) still compares against the plain captions.
Pass: both runs train without encoder reloads. Fail: paste the console lines.

## 6d. Auto caption
On a dataset where some images have no caption: `--auto-caption --set autocaption.limit=5` first, then without the limit.
- Needs `transformers` installed and network once for the model. Task lines "Loading the caption model" and "Captioning"; a `captions:` summary line; the model memory is gone before training (peak not higher than without it).
- Rerun: "0 captioned ... kept as they were": nothing replaced. Captions look sane in the dataset (and training prompts use them).
Pass: captions written, resumed, no memory left behind. Fail: paste the console lines.

## 6e. Gradient alignment
With test 7's probe: add `--set trainer.probe_alignment=true --set trainer.probe_items=4`.
- A `gradient alignment:` line after each probe line with `norm_*`, `share_*`, `self_*`, `cos_*`, `align_*`; `self_*` well above 0 (if near 0, raise `probe_items`).
- Extra time per probe is a few seconds; no new memory peak above the training peak.
Pass: values printed, cosines within -1..1, shares add to 1.

## 17. Mirror padding: compare two LoRAs
Two full runs with the same seed, dataset and settings (packing on, `--max-shapes` as you like): one with `--pad-fill zero`, one with `--pad-fill mirror`.
- Quick check first (20 steps of each): both run, speed and peak are the same.
- Full runs: compare the loss curves (mirror should not be lower by a suspicious amount at high noise) and, more importantly, generations from both LoRAs with the same prompts and seeds. Look at borders/edges, symmetry artifacts, and style strength.
- With a low `--pack` (0.1) the difference should shrink; with a higher one (0.5) it should grow, which tells you padding matters on your data.
Pass: you pick a default; tell me which, and I will set it.

## 18. Epochs
Run `--epochs 2 --save-every-epoch 1 --save-every 100000` on a small dataset.
- Step lines show `epoch 1`, then `epoch 2`; the last step of each says `epoch N end`.
- The run stops after the second epoch (no step limit given); an adapter file is saved at each epoch end (its step number is the epoch's last step).
- `--epochs 3 --steps 10` stops at 10 steps. A cosine schedule node with an empty `total_steps` decays over the whole run.
Pass: all of the above.

## 22. Deterministic run
Same command twice with `--set runtime.deterministic=true --set seed.base=1 --steps 30`, batch size small enough to fit without splitting.
- Compare the two console loss lines and the saved adapters (`sha256sum` or a tensor compare): ideally identical, otherwise the first step where they differ.
- Any `warning` lines about an operation without a deterministic kernel: paste them (those are the remaining sources).
- Speed against the same run without the setting, and whether the peak changed.
Pass: identical losses. If they differ, tell me the first differing step and the warnings; the next suspects are the oneDNN primitive choice and the fused optimizer paths.

## 23. Int8 frozen weights
`pip install xpu-kernels` first (check: `python -m xpu_kernels`). Then the usual short run with `--param adapter.base_precision=int8`, and the same without it.
- A line `int8 base weights (xpu_kernels): N layers, X MB saved` after "Converting the frozen weights to int8"; with `--param adapter.int8_backend=torch` it must run too (very slow; 5 steps are enough) with a warning.
- Peak memory against the native run (expected about 2 GB less at the same batch), and the step time (expected about parity).
- Loss over 100 steps against native: close but not equal; if it diverges or turns to NaN, tell me the step.
- With `--set trainer.graph=true` (graph replay) and with checkpointing on: both must run.
- `--param adapter.kind=dora` with int8: the layers stay native (the summary line says so).
Pass: runs, less memory, speed within a few percent, loss tracks native. Fail: paste the lines and the traceback.

Tested. 
Default:
"Converting the frozen weights to int8 ...
  Converting the frozen weights to int8: 7.6 s
int8 base weights (xpu_kernels): 652 layers, 2059 MB saved; 10 left as they were (640 input features is not a multiple of 256); 81 left as they were (under 1,048,576 parameters)"
Torch:
"Converting the frozen weights to int8 ...
int8 base weights (torch): 652 layers, 2059 MB saved; 10 left as they were (640 input features is not a multiple of 256); 81 left as they were (under 1,048,576 parameters)
[warning] {"node": "n_kkkkkk", "message": "int8 runs on the plain-torch version, several times slower than the kernels (pip install xpu-kernels)"}
"

Speed is ~1500ms/step with default backend, and ~2000ms with torch. Still not good enough. Should be inspected more closely.
With "--set trainer.graph=true" speed is around 1450ms/step. And after step 3 these lines appeared: "B580-diffusion-training/next/families/general/ldm_unet/attention.py:21: UserWarning: SDPA Memory Efficient Attention backend is not supported on XPU, falling back to math backend. (Triggered internally at /pytorch/aten/src/ATen/native/mkldnn/xpu/Attention.cpp:225.)
  out = F.scaled_dot_product_attention(split(q), split(k), split(v))
"

About dora - there is another problem with it: "run: failed: ValueError: Save Adapter: the kohya format cannot hold dora weights; pick another format "

## 24. Memory budget retest (one margin)
The four stacked margins are replaced by `memory.margin_mb` (default 512): allocator cap = limit - outside memory - margin, used by everything. Console lines `memory budget: ...` show the arithmetic.
- `--set memory.vram_max_mb=10240` (batch 3, 512) with `--set trainer.tune_checkpointing=true --param trainer.checkpoint_by=measured`, with and without `--param adapter.base_precision=int8`: the share/placement should now use most of the room (device use close to 10240 minus the margin, not 8.1 GB); paste the `memory budget` lines and the chosen share.
- `--set memory.vram_max_mb=9000`, batch 2, 512, int8: must start (the text encoders load) and train; if it refuses, the message now names the figures.
- Expandable segments: the `limit` line prints `allocator config: ...`. Run the same short run with `PYTORCH_ALLOC_CONF=expandable_segments:True` and `PYTORCH_ALLOC_CONF=expandable_segments:False` exported, native weights, and tell me the peaks (csv `reserved_mb`, `device_used_mb`) and the config line. If both match, the setting has no effect on this workload any more and the earlier 8.3 -> 7.1 GB was something else.
Pass: more of the card used than before at the same limit, no out of memory, the 9000 case runs.

## 25. Int8 speed after the backward change
`pip install -U xpu-kernels` (0.3 or newer: the fused weight dequantizer; older versions are refused). Then:
- `python scripts/bench_int8_layer.py --batch 3 --size 512`: paste the table (both gradient paths, the dequantizer, the dense and int8 input gradient).
- The short run of test 23 with `--set trainer.graph=true` (int8 only pays with graph replay), once with `--param adapter.int8_grad=dense` and once with `int8`: step time, peak, and the loss over 100 steps against native.
- Same with `--param trainer.checkpoint_by=measured --param trainer.tune_checkpointing=true` and graph on: does the placement still choose sensibly? (It measures eagerly; the kernels' measurements say the recompute is a third of a captured step, so it matters more under graph replay.)
Pass: int8 with graph within about 15% of native graph; tell me which gradient path you prefer.
