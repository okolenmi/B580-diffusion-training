# To test on the B580

Things written in the sandbox (no XPU) that need hardware. Delete an entry once it passed; if one fails, keep it
and paste the console lines and the `--memory-log` csv. Feature background: `next/MANUAL.md`.

Common prefix used below (use your usual paths and dataset; 768, batch 2 as in earlier runs):

```
python -m next.run sdxl-lora --models <models> --datasets <datasets> --file <checkpoint> --dataset <name> \
  --output <out> --resolution 768 --batch 2 --steps 200 --memory-log run.csv
```

Run each test from the same start (fresh process). Unless said otherwise only the listed flag changes.

## 1. Baseline with the current defaults
Run the prefix as is.
- `data:` line: shapes <= 16, padding share printed, largest unit printed.
- Step lines mark `new` only a handful of times in the first epoch, never late.
- Ledger table has `ours` and `other` rows; `other` roughly 760-800 MB, flat.
- `cache_returned` events appear when the cache grows; none of them reports an error.
Pass: no crash, peak stable after the first epoch. Paste the `data:` line if padding share looks high.

## 2. Batch area with batch 3 and 4
Add `--batch-area 1.5` and run once with `--batch 4`, once with `--batch 3`.
- Step lines: samples up to 96x144 latents (768 base, 1.5x) show batch 4 (or 3); 96x148 shows 3 (or 2); a 2x
  shape shows 3 (or 2); a 3x shape shows 2 (or 1).
- Peak allocated stays about the same across shapes (compare with `--batch 4` without the flag, which should spike on big shapes).
Pass: batch column follows the rule above and the peak is flat.

## 3. Shape count vs padding
Run with `--max-shapes 8`, then `--max-shapes 49`, then `--max-shapes 0`.
- Count of `new` markers and total seconds of the first epoch go down with fewer shapes.
- `data:` padding share goes up with fewer shapes.
Pass: a table of (max_shapes, first-epoch time, padding share, peak) with the trend above. Tell me which is best for your set.

## 4. Auto checkpoint share
Add `--param trainer.tune_checkpointing=true`, and again with `--set memory.vram_max_mb=10240`.
- `checkpoint_share` event before step 1 with the chosen share and the measured peak per tried share.
- Step time below the baseline run, peak below the budget.
- No OOM retry events in the following steps.
Pass: faster steps than test 1 with the peak under the budget. Fail if peak exceeds the budget or the share is always 1.0.

## 5. Time balance
Run `--time-balance normalize`, then `speed`, then `dro` (100+ steps each; weights start after 10 reports per third).
- Step lines show weights per third, all within 0.25-4.
- Loss does not diverge compared with test 1.
Pass: weights appear and training is stable. Quality judgement is yours: compare sample outputs against test 1.

## 6. Balanced timestep sampling
Add `--balance-sampling 1` (with and without `--time-balance dro`).
- No crash in the batch worker; the share of steps sampling the lagging third (see `loss_t_*` presence on step lines) grows as it lags.
Pass: runs 200 steps cleanly.

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
