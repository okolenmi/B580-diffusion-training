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

## 5. Time balance (retest speed and sampling)
Done: normalize holds [0.25/0.43/2.39] (the smallest-loss third sits at the clip; as designed, a strong setting), dro drifts gently, balance-sampling with dro runs.
Found: `speed` ratcheted to the clip ([1.12/1.14/0.74] -> [2.50/0.26/0.25]) because nothing pulled weights back. It now has a `leak`
(`--param balance.leak=0.05`, default) that forgets a share of the weight at each report. Sampling alone showed nothing on the step line;
it now prints `draws a/b/c` (the chance of each third).
Rerun `--time-balance speed` (100+ steps) and `--balance-sampling 1` alone.
- Expect: speed weights wander but stay moderate (not pinned at 2.5/0.25); `draws` appears with sampling and shifts towards the lagging third.
Pass: both. Fail: paste the weight/draw lines.

## 6. Measured checkpoint placement
Add `--param trainer.tune_checkpointing=true --param trainer.checkpoint_by=measured` (batch 3, 512 as in test 4), then the same with `--set memory.vram_max_mb=10240`.
- Event `checkpoint_plan`: how many blocks stay checkpointed and the list of kept blocks with `mb` and `ms` (real, non-zero numbers; residual blocks may appear).
- Step time versus `checkpoint_by=share` (and versus no tuning); peak within the budget; no crash before step 1.
- If every block is reported checkpointed, paste the `cost` event and the plan: the profile saw no activation memory.
Pass: trains, plan has plausible figures, steps not slower than with `share`. Fail: paste console lines and the csv.

## 7. Probe
Add `--param trainer.probe_every=50`.
- Probe line every 50 steps with `probe_rel_t_*`; the first record is about 1.0 (reference).
- Note whether probe items cause `new` shape markers (they run as batch 1).
- Extra time per probe is small.
Pass: values printed, no extra memory peak above the training peak.

## 8. Conditioning store
Run with `--persist-conditioning` twice in a row.
- First run writes the store (task line); second run prints no encoder loading and starts training sooner.
Pass: second start is clearly faster and the loss at step 1 matches the first run's.

## 9. Cold latent cache
Delete the dataset's latent cache (not the images) and run.
- Task lines with progress; total time (was ~108 s before). Padding images look right.
Pass: finishes; paste the total time.

## 10. Single-shape control
Use a small dataset where all images share one shape and run 200 steps.
- "Outside allocator (ours)" in the ledger stays flat after step ~5. If it still grows, shape count is not the cause.
Pass: flat. Fail: paste the csv.

## 11. Budget cap and planner
Run with `--set memory.vram_max_mb=9216` (tight on purpose).
- `allocator_cap` event with why/ours/other/margin; planner splits big batches (chunk markers on step lines); at most one OOM retry, then training goes on.
Pass: training survives without a crash. Fail: paste the events.

## 12. Optimizer variants
Run `--fused`, then `--param optimizer.group_size=1`, then `--param optimizer.state_precision=int8_blockwise`.
- Peak and step time for each versus test 1; loss similar.
Pass: all train; report peak/step time.

## 13. Allocator mode (experiment)
`PYTORCH_ALLOC_CONF=expandable_segments:True` before the prefix.
- Either an allocator warning (not supported: stop, nothing to fix) or lower `split_free_mb`/`segments` in the csv.
Pass: tell me which.

## 14. Settings that became real
Run the prefix with `--set seed.base=5`, then twice with the same value, then with `--set conditioning.prewarm=false`.
- Same `seed.base` twice: identical first-epoch batch order and step-1 loss; another base: different order.
- `prewarm=false`: no "Encoding prompts" task before training; the encoders run on first use (first steps slower), training still works with `release_encoders` on.
- `--param trainer.timing=false --set runtime.profile=true` prints the per-phase timings.
Pass: all three behave as described.

## 15. Graph replay
Add `--graph` (batch 2, 200 steps), then the same run without it for comparison; a second run with `--max-shapes 4`.
- Console prints `[graph] captured shape (...)` after about 4 sightings of a shape, and steps show `graphs N (R replays)`.
- Replayed steps are clearly faster than eager steps of the same shape (old tree: ~2x for forward+backward).
- Loss curves of the two runs agree closely (old parity: max parameter gap ~3e-3 after 200 steps; the attention kernel differs from eager, so not identical).
- Peak/`device_used` rises by the graph pool (0.5-1.5 GB); check it stays under the card with your other settings; if it does not, lower `trainer.graph_shapes` or drop `--graph`.
- Failure cases worth reporting: `[graph] capture of shape ... failed (...)` (note the message), a loss frozen at one value (outputs not refreshed), a crash at step 4-5.
Pass: faster steps, similar loss, no crash. `--fused` or dropout with `--graph` must refuse with a clear message.

## 16. Batch packing
Use a large batch (`--batch 4`, or bigger if the card allows) with `--batch-area 1.5`. Packing is on by default (0.3): run once with `--pack 0` and once as is.
- `data:` line: `short batches` drops (and `packed` is above 0); `padding mean` rises a little.
- Fewer `new` shape markers over the first epoch, so the first epoch is faster; peak stays as before (the batch-area limit still holds).
- Try `--pack 0.15` and `--pack 0.5` to see how the number of short batches and the padding move; tell me which you prefer.
- Check samples of packed batches in a probe or preview if you can: padding should be noise, not visible in the loss (loss values similar to the run without packing).
Pass: fewer short batches and a faster first epoch, with the same peak and similar loss.

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
