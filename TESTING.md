# To test on the B580

Things written in the sandbox (no XPU) that need hardware. Delete an entry once it passed; if one fails, keep it
and paste the console lines. Feature background: `next/MANUAL.md`.

## Everything that can run unattended: one command

```
python -m next.accept run --out accept --models <models folder> --datasets <datasets folder> --dataset <dataset name> --file <checkpoint file name> \
  --output <output folder> --batch 2 --resolution 512 -- --param buckets.max_shapes=4
```

Runs six suites in separate processes (about 34 short runs, an hour or two; finished scenarios are kept, so rerun the same command after a stop) and writes
`accept/verdict.md`: a pass/fail row per check with the reason, LOOK for what needs your eyes. Send `accept/verdict.md` (and the `accept/matrix/*/run.log` of
anything marked FAIL). `--suites features` (or `speed`, `memory`, `nf4`, `determinism`, `allocator`) runs one suite; `python -m next.accept list` shows every check.

| suite | covers | what it checks |
|---|---|---|
| speed | int8 and native, eager and graph, both int8 gradient paths, attention modes, LoRA in bf16 | every run finishes with a finite loss, int8 saves memory and stays near native graph speed, graph decisions, attention modes beat `math`, losses close |
| memory | one-margin budget at 10240 and 9000 | stays under the limit, uses most of the room, prints its budget |
| nf4 | NF4 frozen weights | runs, saves memory, loss near int8's |
| determinism | two runs with the same seed | identical losses and adapter tensors |
| allocator | expandable segments on and off | the peaks side by side |
| features | probe and gradient alignment, blend, a 9216 cap without prewarm, three `--set`s at once, a typo in `--set`, mirror against zero padding, int8 with the torch backend, int8 with DoRA | each runs and prints what its item expects |

For speed numbers with power and a kernel profile (not pass/fail): `python -m next.perf hwcheck --out perf-bundle ...` as in `next/manual/performance.md`.

## Still manual

Experiments of the lab can be queued: `python -m next.lab run long.json loraplus.json aligned2.json ema.json more.json loraplus2.json` runs them one after the other. Training states are written every 200 steps, so after a power cut the same command skips what is finished and continues the arm that was cut (TESTING 58 checks this once).

## 6b. Merge Adapter
Add a `Merge Adapter` node (denoiser from Load Model -> Merge Adapter -> Inject Adapter) with a kohya LoRA you have, `strength` 1 (the CLI starters have no switch for it; edit the graph).
- Task line "Merging <file>"; training starts and the step-1 loss differs from the unmerged run; a LoRA of another family or a wrong key format stops with "nothing ... matches".
- Strength 0 gives the same step-1 loss as no merge node.
Pass: all three. Fail: paste the message.

## 6d. Auto caption
On a dataset where some images have no caption: `--auto-caption --set autocaption.limit=5` first, then without the limit.
- Needs `transformers` installed and network once for the model. Task lines "Loading the caption model" and "Captioning"; a `captions:` summary line; the model memory is gone before training (peak not higher than without it).
- Rerun: "0 captioned ... kept as they were": nothing replaced. Captions look sane in the dataset (and training prompts use them).
Pass: captions written, resumed, no memory left behind. Fail: paste the console lines.

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

## 21. Experiment: kernel compile cache between runs
You saw a default run after a graph run start with almost no new-shape penalty, and the next run slow again. The driver keeps a compiled-kernel cache on disk that may fill up and evict.
Try `NEO_CACHE_MAX_SIZE=4294967296` (4 GB) before two default runs in a row, and compare the first-sighting step times (`[... new]` markers) of the second run with a run without it.
Pass: tell me whether first sightings of shapes are faster on the second run. No fix needed either way.

## 29. NF4 frozen weights
First, no training: `python -m next.accept quant-probe --models <models> --file <checkpoint> --precision nf4 int8 --resolution 512 --out probe` (needs the card; each precision loads the checkpoint again). Send `probe/quant-nf4.md` and `quant-int8.md`: the whole-network error and the table by region show where the quantization hurts (the worst block is named). Then training: `--param adapter.base_precision=nf4` (once with `--param adapter.nf4_double_quant=true`), eager and `--set trainer.graph=true`, short run with `--perf`: step time, peak memory, the `quantization` line (layers and MB saved), and the loss over 100 steps against native and int8.
Known risks: the fused dequantizer's first call verifies itself (a few seconds, once); `xpu-kernels` older than 0.3 has no NF4 kernel and `auto` then uses the slow torch chain (a warning says so).
Pass: no out of memory at a batch or resolution that int8 cannot hold; loss within the noise of int8's at the same seed.

## 30. Padding decision by eye (mirror against zero)
`python -m next.accept padding-sheet --images <a few of your dataset images> --canvas 128x128 --count 4 --out sheet.png`, then look at `sheet.png`: per sample, the clean placement, zero padding (the strip is noise alone after noising) and mirror padding (a reflection of the noisy sample). Tell me which you would train on; the default stays zero until you say.

## 61. Continue your best adapter (about 1.5 hours)
Put your saved adapter (batch 1, lr 2e-4, 600 steps) into `lab-setup.json` under `loras` as `best`. The template renders it as it is and continues it three ways with a new `--init-adapter` option: 300 more steps at 2e-4, 300 at 5e-5, 600 at 2e-4 (fresh optimizer, 50 step warmup, no region ladder).
```
python -m next.lab template continue-best > more.json
python -m next.lab run more.json
```
Send the zip. Does more training improve it or damage it? Compare each arm with `best` on the same sheet and check the holdout column of the per-timestep table (above 1 = worse than the base model). If the continuation fails to start with a message about rank or targets, the adapter was trained with other settings than the template's (rank 16 default, all default targets); tell me which.

## 62. Queue run and the cache warning (check while running the above)
Run two files in one command (for example `python -m next.lab run aligned2.json more.json`). Each file now runs in its own process; the second file's training should start without the out-of-memory error that chained runs gave. Send the console of any failure. Also: the `oneDNN primitive cache is evicting shapes` warning should no longer appear (the cache floor is 4096 and counts four batch sizes per shape); tell me if it does.

## 65. LoRA+ ratios 2, 4, 8, 12 (about 3 hours for five arms; pick with `--arms`)
Both 4 and 16 looked good in the first test; this narrows it down on the same recipe, with the baseline rerun for one sheet.
```
python -m next.lab template lora-plus-values > loraplus2.json
python -m next.lab run loraplus2.json
```
Send the zip. Which ratio gives the best textures without painted or unstable pictures, and is there a visible step between neighbours or only noise?

## 66. After the cleaning: a much lower learning rate (about 1.5 hours)
Follow-up of 64: the cleaned and frozen 15% plus a cleared 85% that trains at 2e-5 and 4e-6 instead of 2e-4 (your overshoot idea). Same cleaning settings as in 64 (the cleaning is repeated in each arm, about 11 minutes).
```
python -m next.lab template clean-lr > cleanlr.json
python -m next.lab run cleanlr.json
```
Send the zip. Compare with `best` and `element-15-only` on the same sheet, and the holdout column of the per-timestep table (a lower ratio than `best` = better than the base model there).

## 67. The cleaner with other anchor weights (about 1 hour)
Result of 64: anchor 5 removed nearly all of the dataset's effect, no anchor kept most of it. The cleaned-only arms with anchor 0.5, 1 and 2 (element) and 1 (component) show where it changes.
```
python -m next.lab template clean-anchor > cleananchor.json
python -m next.lab run cleananchor.json
```
Send the zip. Look at the portrait and red house pictures against `best` (how much of the base model's look comes back) and at the empty prompt (how much of the dataset's look stays).

## 68. Magnitude cleaning variants (about 1.5 hours; no training in the arms)
Your verdict on 64: magnitude (the largest 15% of the weights) is the best of the cleaners. Variants: 8% and 30%; 15% counted per layer instead of over the whole adapter (`--clean-scope layer`); ranking by what a weight contributes to the layer's change (`--clean-mode product`); scaling every layer's change back to its old size after the cut (`--clean-rescale`); and the largest singular directions per layer (`--clean-mode singular`).
```
python -m next.lab template clean-magnitude > cleanmag.json
python -m next.lab run cleanmag.json
```
Send the zip. Which keeps the effect of `best` with the fewest issues? Does `rescale` give back the power without bringing the problems back?

## 69. Which parts of the adapter matter (about 1.5 hours; no training)
`best` with one part or region removed completely (feed-forward, attn1, attn2, projections, input, middle or output blocks) with 99% of the rest kept.
```
python -m next.lab template clean-parts > cleanparts.json
python -m next.lab run cleanparts.json
```
Send the zip. Which removal changes the pictures most, and which one removes an issue without losing the effect? That tells which parts to protect or to train less.

## 70. Train without cross-attention and without the sensitive blocks (about 2.5 hours for four trained arms; pick with `--arms`)
Result of 69: removing attn2, the middle block or the input blocks from your best adapter did not change the loss at all and the pictures got cleaner. This trains with those parts excluded from the start (your best recipe, 600 steps): no attn2; no middle block; both; and both plus input blocks 6-8 and output blocks 0-2. `best` is rendered for the same sheet.
```
python -m next.lab template targets > targets.json
python -m next.lab run targets.json
```
Send the zip. Do the cleaner pictures hold up when the parts are never trained rather than cut afterwards? The console prints the number of adapted layers; fewer layers should also mean a little faster steps and less memory.

## 71. Staged training in one run (about 5 hours for four trained arms; pick with `--arms`)
Your plan: train, clean to 8% and freeze, train the rest at a lower rate, clean to 15%, and so on, all in one process. Arms: `8-15-30` (600 steps, then 300 at 5e-5 and 300 at 2e-5, cleaned to 8%, 15%, 30%), `8-15`, `8-slow` (cleaned to 8%, then 600 steps at 2e-5) and `15-30`. `best` is rendered for the same sheet.
```
python -m next.lab template stages > stages.json
python -m next.lab run stages.json
```
Check on the way: the console prints a "cleaning before step 600 (magnitude)" line at every boundary and the loss does not jump up after it (a short rise while the rate ramps is fine); `<output>/clean` holds one `lora_cleaned_stepN.safetensors` per cleaning. Send the zip. Do the later stages add effect without the problems coming back, and which share and rate are enough? The cleaned adapter of the last cleaning is a better candidate than the last saved step.
You can also run it by hand: `--stage 600:0.08 --stage 300@5e-5:0.15 --stage 300@2e-5:0.3` with your usual arguments (`--stage` replaces `--steps`).

## 72. A learning rate per part of the network (about 3 hours for four trained arms; pick with `--arms`)
Your estimate: the middle block is the most sensitive, input blocks 6-8 and output blocks 0-2 the second tier. Arms: `mild`, `tiers`, `strong` (lower and lower rates for those parts, `--lr-region`) and `middle-only`. `best` is rendered for the same sheet.
```
python -m next.lab template block-rates > blockrates.json
python -m next.lab run blockrates.json
```
Send the zip. Does the loss stay the same while the pictures get cleaner? No warning about an unmatched region should appear at the start of a run (a misspelt prefix is reported there).

## 75. Is a run that was stopped and started again the same run? (about 3 hours: six arms of 600 steps, the lab stops twice)
The anchor run's baseline was cut and continued from step 400 and gave the best pictures yet. The CPU says a cut run is identical to an uninterrupted one to the last bit (`check_resume`); the first card try (200 steps, cut at 100) agreed for the deterministic arms and differed for the ordinary one, but was too short to compare with the baseline. This version has the baseline's recipe and length.
```
python -m next.lab template resume-proof > resume.json
python -m next.lab run resume.json
```
The lab trains `full-a`, then trains `cut-400` up to step 400 and **stops itself** (state saved, nothing left running). Start the same `python -m next.lab run resume.json` again: it continues `cut-400` as a new lab and a new process (close the window first; reboot if you want it as hard as the power cut). It stops once more at `cut-det-400`; start it again. After that the rest trains and the pictures are made. Arms: `full-a`, `full-b` and `cut-400` are ordinary runs (`full-a` repeats the anchor baseline); `full-det-a`, `full-det-b` and `cut-det-400` are the same with deterministic kernels. Drop `full-b` or the `-det` arms from the file if time is short, but keep `full-det-b`: without it a difference of `cut-det-400` proves nothing.
Send the zip. How to read it: the report's table of adapters against the first arm. Deterministic arms must read `yes` between themselves (the two full ones first: if they differ, the kernels are not deterministic on the card and the cut test says nothing). If `cut-400` is no farther from `full-a` than `full-b` is, a restart changes nothing the ordinary noise does not. Also look at the pictures of the ordinary and the deterministic arms side by side on every prompt: in the first try the deterministic ones looked better on most images, which, if it holds at 600 steps, says the kernels' noise costs quality.

## 28. Chunked attention, memory side
`--set trainer.graph=true --param trainer.graph_attention=chunked_bf16` against `math_bf16` at 1024 px: the step peak should be lower with `chunked_bf16` (the speed and loss sides are in the `speed` suite).

## 74. Backend run on the new graph document
Take a graph document (format 3) that `python -m next.run graph graph.json` already runs for you and post it to the backend as `{"graph": <the document>}` at `POST /api/v1/graphs/run` (validate first at `/validate`). It should be admitted, the child should start on the XPU, and `GET /api/v1/graphs/executions/{id}` should list the nodes by uid. The admitted budget replaces `memory.vram_max_mb` of the graph. Run the same graph a second time: the second start should be admitted against the first run's measured peak (the execution's memory block shows `demand_source` as observed). Report anything the child prints about a missing folder or an unknown service: it builds its model and dataset folders from the paths its parent hands down.

## 76. The new graph editor (`/graph2`)
Start the backend and open `http://127.0.0.1:<port>/graph2` (the old editor stays at `/graph`). Check, in this order: the palette lists every node class with no "Modules that failed to load" entry; double-click the canvas and add a Load Model node, pick a checkpoint in its file picker (the list must be the files of the node's scope); add a dataset node and pick a dataset; drag from an output socket to an input socket (compatible inputs light green while dragging, incompatible ones fade); drop a wire on empty canvas and add a node from the list (it must arrive already connected); Ctrl+C / Ctrl+V a few nodes; reload the page (the canvas must come back); Save to the library and Load it again. Start with New from starter: pick your checkpoint and a dataset, set an output folder, Create (15 nodes in columns, the whole graph in view, no red unknown nodes); Validate it and Run it for a few steps. Also: double-click the canvas (the add-node menu must open at the pointer, scroll with the wheel and add on click), drag a wire from any output and drop it on empty canvas (the list must show nodes that fit and add the node already connected), select nodes and Ctrl+C / Ctrl+V; value fields show their socket only when the pointer is over their row or a wire is being dragged. Node cards have tabs like a browser's: Main, Optional (tick an optional input to give it a socket; a wired one stays ticked) and Addons (enable switch, parameters, remove, "+ Add addon"; open it on a Noiser, Batcher, schedule or Trainer); a wire to an addon parameter shows as a circle with a number on the card's left edge while the Addons tab is closed, and clicking it opens the tab. Drag a wire onto a node's tab and hold: the tab opens, and closes again when you let go. Drop a wire from an input on empty canvas: the list has a `main` section and a `may have this connector` section (Load Model there; it is added unconnected). Dataset pickers (Images, Captions, Latents nodes, starter dialog) list image libraries only (e.g. `my_set`), so a picked name always opens in the run; a true/false setting is a bare checkbox. Number fields: the spinner arrows step by the number's size (0.0002 to 0.0003) and `2e-4` becomes `0.0002`. The first page load should not wait several seconds for the node list. Constant LR and Cosine LR hold the learning rate; the Optimizer nodes no longer have one. Also: write a graph with `python -m next.run sdxl-lora ... --dump graph.json`, press Import file and pick it (the nodes must arrive in columns left to right with no overlap; Arrange redoes it once the cards have their real size), then Export and import the exported file again. Then build the graph you normally train with, press Validate (issues must name nodes you can click), watch the Memory panel, and Run: node badges should turn done with a time, Stop must work, and a second Run while one is active must say so. Report anything that looks wrong or feels slow, especially a node whose ports change when a param changes (the card should say "processing" and then update).

## 77. Page styling after the control-skin cleanup
Open every page (Training, Config, Datasets, Templates, Settings, Setup, Help, Monitor, both graph editors) and look at the text fields, selects and checkboxes: they now share one look (dark field, 4px corners, same size per area). Report any control that looks too big, too small or different from its neighbours, and any page where something moved or lost its styling. The old Training and Run page sheets were unused and are deleted.

## 78. Datasets page for image libraries
The Datasets page manages image libraries (what the training nodes open); the old trajectory datasets are gone. It was redesigned as a workbench and checked in a browser against the real backend, so these are the things only your data and eyes can judge.
(1) List: each dataset is a card with a mosaic of its newest pictures, a caption-completeness bar and its sets/caches; the dashed first card creates one. Your old `my_set` should be there. (2) Open one, "Import images" with the path of a folder of your pictures (with `name.txt` captions beside them): progress shows in the dialog and, after closing it, as a pill in the header. Try a large folder and say how long it takes and whether the page stays usable. (3) The mosaic keeps each picture's proportions; "Balanced" (default) gives tall and wide pictures about the same area, "Rows" is the plain justified-row grid. Switch between them, drag the size slider and resize the window: is relayout quick enough, does it jump, is the balance what you wanted? Scroll through a few thousand pictures: does loading more feel smooth? (4) Click a picture: the inspector shows it with a caption editor that saves by itself (watch "Saved"); switch text/tags/desc; Shift-click a range, Ctrl-click to add; arrows move, Space opens the full-size viewer (arrows move inside it), Esc clears. (5) With several selected: "Add at the start" a trigger word (the preview shows the result on the first picture), "Add" to a set, then drag a selection onto a set tab. (6) "Without text caption" filter, search, and the Library tab: do shapes, completeness, repeated captions and top tags match what you know of the set? Clicking a tag searches for it. (7) Remove from dataset asks first and leaves your files alone; Delete dataset likewise. (8) After a run, a latent cache per VAE shows in the Library tab with its size and can be deleted. Say anything that feels clumsy, slow or ugly: this page is meant to be finished, not a first cut.

## Reading a bundle automatically
`python -m next.accept judge <folder with matrix/> speed` judges a bundle that came back from `python -m next.perf hwcheck` without running anything.
