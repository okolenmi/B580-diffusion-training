# To test on the B580

Things written in the sandbox (no XPU) that need hardware. Delete an entry once it passed; if one fails, keep it
and paste the console lines and the `--memory-log` csv. Feature background: `next/MANUAL.md`.

Common prefix used below (use your usual paths and dataset; 768, batch 2 as in earlier runs):

```
python -m next.run sdxl-lora --models <models> --datasets <datasets> --file <checkpoint> --dataset <name> \
  --output <out> --resolution 768 --batch 2 --steps 200 --memory-log run.csv
```

Run each test from the same start (fresh process). Unless said otherwise only the listed flag changes.

## 4. Auto checkpoint share, with a tight budget (retest)
Your run: unbounded budget kept share 1.0 (10.3 GB at batch 3, 512); batch 1 at 512 chose 0.75 and was faster (1040 -> 970 ms).
With `--set memory.vram_max_mb=10240` it died before step 1: the tuner tried a lower share, which stores more activations, and that
measuring run hit the cap. Fixed: a share whose measuring run runs out of memory now counts as too expensive.
Rerun the 10240 case (batch 3, 512).
- Expect: no crash before step 1; `checkpoint_share` event (probably 1.0 or 0.75); the step lines then run, with chunk markers if the planner splits batches.
Pass: it trains. Fail: paste the console lines and the csv.

Tested... But the way this parameters set is confusing. One is through param another through set and Manual don't say it (or maybe it says somewhere, but it's so big I wouldn't read it whole). Overall Maunual is badly structured for a human reader.

Everything is ok. Limit works (can't enable checkpointing with such limit), but not sure it's vorrect as it holds everything under 8.5GB (nvtop monitor) when it is set to 10240. 

## 5. Time balance (retest speed and sampling)
Done: normalize holds [0.25/0.43/2.39] (the smallest-loss third sits at the clip; as designed, a strong setting), dro drifts gently, balance-sampling with dro runs.
Found: `speed` ratcheted to the clip ([1.12/1.14/0.74] -> [2.50/0.26/0.25]) because nothing pulled weights back. It now has a `leak`
(`--param balance.leak=0.05`, default) that forgets a share of the weight at each report. Sampling alone showed nothing on the step line;
it now prints `draws a/b/c` (the chance of each third).
Rerun `--time-balance speed` (100+ steps) and `--balance-sampling 1` alone.
- Expect: speed weights wander but stay moderate (not pinned at 2.5/0.25); `draws` appears with sampling and shifts towards the lagging third.
Pass: both. Fail: paste the weight/draw lines.

Tested. Both move towards equality. Speed around 1.0, --balance-sampling (draws) near 0.33

## 6. Measured checkpoint placement
Add `--param trainer.tune_checkpointing=true --param trainer.checkpoint_by=measured` (batch 3, 512 as in test 4), then the same with `--set memory.vram_max_mb=10240`.
- Event `checkpoint_plan`: how many blocks stay checkpointed and the list of kept blocks with `mb` and `ms` (real, non-zero numbers; residual blocks may appear).
- Step time versus `checkpoint_by=share` (and versus no tuning); peak within the budget; no crash before step 1.
- If every block is reported checkpointed, paste the `cost` event and the plan: the profile saw no activation memory.
Pass: trains, plan has plausible figures, steps not slower than with `share`. Fail: paste console lines and the csv.

Tested: first one gives ~930 (1010ms without it) ms step on batch size 3 512x - VRAM less than 10.5GB in average from trainer process only (up to ~8.5GB without it). Without measured parameter share automatically becomes 1.0, so improvement with current parameters may be impossible (to 0.75) or internal limits are too strict.

With memory limit it works as well, but slightly slower - I have "event": "cache_returned" after each step. Speed decreasing to 1130ms in average. With 1 extra gb of limit I have ~950ms/step. Memory is under 9.5GB.

## 6b. Merge Adapter
Add a `Merge Adapter` node (denoiser from Load Model -> Merge Adapter -> Inject Adapter) with a kohya LoRA you have, `strength` 1 (the CLI starters have no switch for it; edit the graph).
- Task line "Merging <file>"; training starts and the step-1 loss differs from the unmerged run; a LoRA of another family or a wrong key format stops with "nothing ... matches".
- Strength 0 gives the same step-1 loss as no merge node.
Pass: all three. Fail: paste the message.

## 6c. Conditioning blend
Add `--param blend.strength_max=0.5` (min stays 0), then `--param blend.strength_min=0.5 --param blend.strength_max=0.5`.
- Trains; with `--persist-conditioning` and prewarm on, no encoder loading after the first epoch starts (the empty prompt is encoded up front).
- Fixed 0.5: loss is higher at the start than unblended and the step time is unchanged; the probe (`trainer.probe_every`) still compares against the plain captions.
Pass: both runs train without encoder reloads. Fail: paste the console lines.

## 6d. Auto caption
On a dataset where some images have no caption: `--auto-caption --param autocaption.limit=5` first, then without the limit.
- Needs `transformers` installed and network once for the model. Task lines "Loading the caption model" and "Captioning"; a `captions:` summary line; the model memory is gone before training (peak not higher than without it).
- Rerun: "0 captioned ... kept as they were": nothing replaced. Captions look sane in the dataset (and training prompts use them).
Pass: captions written, resumed, no memory left behind. Fail: paste the console lines.

## 6e. Gradient alignment
With test 7's probe: add `--param trainer.probe_alignment=true --param trainer.probe_items=4`.
- A `gradient alignment:` line after each probe line with `norm_*`, `share_*`, `self_*`, `cos_*`, `align_*`; `self_*` well above 0 (if near 0, raise `probe_items`).
- Extra time per probe is a few seconds; no new memory peak above the training peak.
Pass: values printed, cosines within -1..1, shares add to 1.

## 7. Probe
Add `--param trainer.probe_every=50`.
- Probe line every 50 steps with `probe_rel_t_*`; the first record is about 1.0 (reference).
- Note whether probe items cause `new` shape markers (they run as batch 1).
- Extra time per probe is small.
Pass: values printed, no extra memory peak above the training peak.

Tested. Fells like something is wrong. There are only one line after first step, then default training. That one line: "probe: adapter/base loss by third 1.000/1.000/1.000, worst 1.000 (above 1: worse than the base model)"

## 8. Conditioning store
Run with `--persist-conditioning` twice in a row.
- First run writes the store (task line); second run prints no encoder loading and starts training sooner.
Pass: second start is clearly faster and the loss at step 1 matches the first run's.

Tested. Works fine. Text encoding was skipped in second run.

## 9. Cold latent cache
Delete the dataset's latent cache (not the images) and run.
- Task lines with progress; total time (was ~108 s before). Padding images look right.
Pass: finishes; paste the total time.

Tested. 201 512x Latents were encoded in 62 seconds. Not sure about padding - I'm testing it only in console now (16 padded samples overall if you are talking about this). VRAM consumption jumped sometimes (two sharp +~1GB jumps from ~6GB to ~8GB overall).

## 10. Single-shape control
Use a small dataset where all images share one shape and run 200 steps.
- "Outside allocator (ours)" in the ledger stays flat after step ~5. If it still grows, shape count is not the cause.
Pass: flat. Fail: paste the csv.

## 11. Budget cap and planner
Run with `--set memory.vram_max_mb=9216` (tight on purpose).
- `allocator_cap` event with why/ours/other/margin; planner splits big batches (chunk markers on step lines); at most one OOM retry, then training goes on.
Pass: training survives without a crash. Fail: paste the events.

Tested. A fail:   "Measuring what a pixel costs: 5.0 s
[  14.0s] DiffusionTrainerNode: 10.4s  allocated 5325, reserved 6418, device 8422 MB
run: failed: RuntimeError: optimizer state is offloaded; reload it before stepping"

## 12. Optimizer variants
Run `--fused`, then `--param optimizer.group_size=1`, then `--param optimizer.state_precision=int8_blockwise`.
- Peak and step time for each versus test 1; loss similar.
Pass: all train; report peak/step time.

Tested.
Fused: ~1200ms/step VRAM is close to what I have without it.
group_size 1: ~1200ms as well, the same VRAM usage (~8.3GB, I guess - this process only)
fused + group_size 1: ~1175ms/step for some reason
all 3 together: fail "fused optimizer needs float32 state"
group_size 1 + int8 - 2250ms/step
int8 alone - 1030ms/step
VRAM - almsot the same. Optimizer is cheap for LoRA. int8 optimizer with rank16 lora is 84mb, gradients 162mb.

## 13. Allocator mode (experiment)
`PYTORCH_ALLOC_CONF=expandable_segments:True` before the prefix.
- Either an allocator warning (not supported: stop, nothing to fix) or lower `split_free_mb`/`segments` in the csv.
Pass: tell me which.

Tested. It's REALLY good improvement. VRAM went from ~8.3 -> 7.1GB (Now LoRA training on 8GB cards is comfortable, I would say...). No speed penalty. No warnings/errors.
136.96,step 71,5833.9,6484.0,6189.3,7952.2,7394.6,5833.9,0.0,0.0,
137.21,step 71,5578.4,6484.0,6189.3,7952.1,7394.6,5578.4,0.0,0.0,
137.47,step 72,5581.6,6484.0,5602.2,7952.1,7394.6,5581.6,0.0,0.0,

## 14. Settings that became real
Run the prefix with `--set seed.base=5`, then twice with the same value, then with `--set conditioning.prewarm=false`.
- Same `seed.base` twice: identical first-epoch batch order and step-1 loss; another base: different order.
- `prewarm=false`: no "Encoding prompts" task before training; the encoders run on first use (first steps slower), training still works with `release_encoders` on.
- `--param trainer.timing=false --set runtime.profile=true` prints the per-phase timings.
Pass: all three behave as described.

Tested. Losses are the same on the same seed (with different encoder loading approaches as well). Or almost the same. Third loss were 0.1064->0.1065->0.1064 (3 runs). Exact behavior may need pytorch --deterministic flag, I think.

About "prewarm: false" - I can't see a line with prompting encoders. Also don't see VRAM spikes. But I have only 1 prompt in that dataset... It may be encoded before first step, so there should be no big VRAM movement.

## 15. Graph replay
Add `--graph` (batch 2, 200 steps), then the same run without it for comparison; a second run with `--max-shapes 4`.
- Console prints `[graph] captured shape (...)` after about 4 sightings of a shape, and steps show `graphs N (R replays)`.
- Replayed steps are clearly faster than eager steps of the same shape (old tree: ~2x for forward+backward).
- Loss curves of the two runs agree closely (old parity: max parameter gap ~3e-3 after 200 steps; the attention kernel differs from eager, so not identical).
- Peak/`device_used` rises by the graph pool (0.5-1.5 GB); check it stays under the card with your other settings; if it does not, lower `trainer.graph_shapes` or drop `--graph`.
- Failure cases worth reporting: `[graph] capture of shape ... failed (...)` (note the message), a loss frozen at one value (outputs not refreshed), a crash at step 4-5.
Pass: faster steps, similar loss, no crash. `--fused` or dropout with `--graph` must refuse with a clear message.

Tested. At the beginning there may be penalty rarely - 1800ms on non-new shapes. But with time situation becomes better. For some reaon it works especially effective for small shapes:
step 149/200  loss 0.0386  544 ms  epoch 2  [1x112x64]  graphs 7 (61 replays)  device 9010 MB (live 5551, step peak 5766, cached 7214)
step 150/200  loss 0.2707  547 ms  epoch 2  [1x112x64]  graphs 7 (62 replays)  device 9010 MB (live 5551, step peak 5766, cached 7214)
step 151/200  loss 0.0461  545 ms  epoch 2  [2x64x64]  graphs 7 (63 replays)  device 9010 MB (live 5551, step peak 5766, cached 7214)
step 152/200  loss 0.2341  550 ms  epoch 2  [2x64x64]  graphs 7 (64 replays)  device 9010 MB (live 5551, step peak 5766, cached 7214)
step 153/200  loss 0.0222  787 ms  epoch 2  [2x92x64]  graphs 7 (65 replays)  device 9010 MB (live 5551, step peak 5766, cached 7214)

Penalty example:
step 177/200  loss 0.2659  1742 ms  epoch 2  [1x128x64]  graphs 8 (77 replays)  device 8675 MB (live 5552, step peak 6307, cached 6836)
Unstable behavior:
step 192/200  loss 0.3402  1040 ms  epoch 2  [1x64x116]  graphs 8 (81 replays)  device 9015 MB (live 5552, step peak 5896, cached 7164)
step 193/200  loss 0.2322  1010 ms  epoch 2  [2x96x64]  graphs 8 (81 replays)  device 9056 MB (live 5552, step peak 6120, cached 7224)
step 194/200  loss 0.1647  961 ms  epoch 2  [2x96x64]  graphs 8 (81 replays)  device 9056 MB (live 5552, step peak 6120, cached 7224)
step 195/200  loss 0.0480  1033 ms  epoch 2  [2x96x64]  graphs 8 (81 replays)  device 9082 MB (live 5552, step peak 6120, cached 7224)
step 196/200  loss 0.0092  1008 ms  epoch 2  [2x96x64]  graphs 8 (81 replays)  device 9074 MB (live 5552, step peak 6120, cached 7224)
step 197/200  loss 0.0120  539 ms  epoch 2  [1x104x64]  graphs 8 (82 replays)  device 9058 MB (live 5552, step peak 5766, cached 7224)
step 198/200  loss 0.0382  511 ms  epoch 2  [1x104x64]  graphs 8 (83 replays)  device 9057 MB (live 5552, step peak 5766, cached 7224)
step 199/200  loss 0.1705  511 ms  epoch 2  [1x104x64]  graphs 8 (84 replays)  device 9057 MB (live 5552, step peak 5766, cached 7224)

VRAM usage here is already with that pytorch environment variable (~+1.5GB).
Loss is the same. Step 40 (0.0979->0.0979)

And... There is a strange thing. After that graph run I started default one and:
step 22/200  loss 0.0230  980 ms  epoch 1  [1x128x64 new]  device 7481 MB (live 5546, step peak 5926, cached 6128)
Almost all new shapes were calculated without penalty? -_- Next run after this was slow again... Maybe shape cache have some sort of lifetime and it may stay between runs. Not sure. Maybe it's something unique for the same seed usage.

## 16. Batch packing
Use a large batch (`--batch 4`, or bigger if the card allows) with `--batch-area 1.5`. Packing is on by default (0.3): run once with `--pack 0` and once as is.
- `data:` line: `short batches` drops (and `packed` is above 0); `padding mean` rises a little.
- Fewer `new` shape markers over the first epoch, so the first epoch is faster; peak stays as before (the batch-area limit still holds).
- Try `--pack 0.15` and `--pack 0.5` to see how the number of short batches and the padding move; tell me which you prefer.
- Check samples of packed batches in a probe or preview if you can: padding should be noise, not visible in the loss (loss values similar to the run without packing).
Pass: fewer short batches and a faster first epoch, with the same peak and similar loss.

Tested. On batch size 8 (it feels like I can even do 32, but speed will be horrible):
default 1.5x -> "data: 204 samples, 72 latent shapes -> 16 padded shapes (16 with short batches) in 28 batches; padding mean 5%, worst 29%; 1 short batches, 28 samples packed up, 0 dropped"

BTW, tested batch size 32. And your prediction is very wrong here: memory: {"event": "split", "samples": 22, "chunks": [11, 11], "predicted_mb": 13337, "budget_mb": 10174} - usage is under 10GB in average with that environment variable. Average speed is 8000ms per step.
data: 204 samples, 72 latent shapes -> 16 padded shapes (8 with short batches) in 8 batches; padding mean 10%, worst 31%; 2 short batches, 148 samples packed up, 0 dropped

Back to batch size 8. pack=0: data: 204 samples, 72 latent shapes -> 16 padded shapes (26 with short batches) in 34 batches; padding mean 3%, worst 14%; 14 short batches, 0 samples packed up, 0 dropped
pack=0.15: data: 204 samples, 72 latent shapes -> 16 padded shapes (17 with short batches) in 29 batches; padding mean 4%, worst 22%; 3 short batches, 28 samples packed up, 0 dropped
pack=0.5: data: 204 samples, 72 latent shapes -> 16 padded shapes (16 with short batches) in 28 batches; padding mean 5%, worst 29%; 1 short batches, 28 samples packed up, 0 dropped

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
