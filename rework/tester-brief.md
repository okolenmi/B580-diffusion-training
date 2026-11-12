# Brief for the AI that runs hardware checks on the owner's PC

You run checks on an Intel Arc B580 (12 GB) for the SDXL LoRA trainer in this repository. You do not develop it. The owner has about 200,000 tokens a day for
you and power cuts can stop a run at any time, so: **cheap, resumable, short reports.**

## Rules
* Read this file and the `TESTING.md` items a mission names; do not read source code unless a mission says so. Never edit repository files.
* Never print a log into the chat. Use `tail -n 30`, `grep -n -i "error\|traceback\|fail"`, `head`. Long runs: start them in the background with output to a
  file (`nohup ... > run.log 2>&1 &`), then check in with a single `sleep 600; tail -n 5 run.log` per check, not a loop of small checks.
* Nothing else may use the card while you measure (browser video, games, ComfyUI). Say so to the owner before starting a mission that needs it.
* Put outputs in a scratch folder `scratch/` beside the repo. Do not delete the owner's models, datasets or outputs. Ask for the paths if you do not know them
  (models folder, datasets folder, a dataset name, a checkpoint file name, an output folder); use the same paths for every mission.
* If a command fails the same way twice, stop that mission and report; do not try ten variations. If a run takes more than twice the time given here, stop it.
* Write one report per mission to `scratch/reports/<date>-<mission>.md`, at most 60 lines: the exact commands, one line per check (`PASS`, `FAIL`, `UNSURE`
  and why), the numbers asked for, and for each failure the first 15 lines of the error. The owner sends the reports to the developer.

## Mission 1: the unattended suite (cheapest, most valuable; 1 to 2 hours, resumable)
Run the one command at the top of `TESTING.md` ("Everything that can run unattended") with the owner's paths. If power is lost, run the same command again:
finished scenarios are kept. Report `accept/verdict.md` (the table as it is) and the `run.log` tail of every FAIL row.

## Mission 2: a real run through the backend (TESTING 74, 82; about 1 hour)
1. Start the server as `run_server.sh` does (`python -m backend.cli`, port printed on start). Check `GET /api/v1/health` (or `/api/v1/graphs/executions`).
2. Get a graph: `python -m next.run sdxl-lora --models M --file F --dataset D --output O --dump scratch/g.json` (use `--help` if an option differs), or
   `POST /api/v1/starters/sdxl-lora`. Set steps to about 150 and resolution 512, batch 2.
3. `POST /api/v1/graphs/validate`, then `POST /api/v1/graphs/run` with `{"graph": <document>, "mode": "parallel"}`. Poll `GET /api/v1/graphs/executions/{id}` every
   30 s until finished. Report: status, per-node timings, the `memory` block (`demand_source`, `reserved_mb`), and whether the adapter file appeared in the output.
4. Run the same graph again: `demand_source` should now be `observed`. Report it.
5. Parallel: from the first run's measured peak P (MB), start one run with `"memory_overrides": {"vram_max_mb": P}` and, while it runs, a second with
   `"mode": "parallel"` and the same override. Report whether the second was admitted or refused (409 `memory_unavailable`; paste its `details`), then repeat the
   second with `"mode": "auto"` and report that it comes back `planned`, and that it starts by itself within about 10 s after the first ends.
   If two runs do run side by side, report each one's step time from the event file against a run alone (`graph_executions/execution_N.events.jsonl`: `step_ms`).
   Stop all runs when done (`POST /api/v1/graphs/executions/{id}/stop`).

## Mission 3: the editor in a browser (TESTING 76, 79, 80, 82, 83; about 1.5 hours)
Needs Playwright (`pip install playwright && playwright install chromium`); if it cannot be installed, skip the mission and say so. Open
`http://127.0.0.1:<port>/graph2`, and for each item in `TESTING.md` (read only items 76, 79, 80, 82, 83) do what it says through Playwright and report PASS/FAIL per
numbered point. Take a screenshot only where a point is a visual judgement (layout, overlap) or fails, at 1200x800, and name the file in the report; do not
describe screenshots at length. Keep one real training running for the points that need one (a 200-step starter).

## Mission 4: the datasets page (TESTING 78; about 30 minutes)
Same method as mission 3 for `http://127.0.0.1:<port>/datasets` with a folder of the owner's pictures. Time the import and the first scroll through the whole mosaic;
report the sizes of the smallest and largest tile in "Balanced" (ratio is enough).

## Mission 5: long experiments (only when the owner asks; hours, token-cheap)
`TESTING.md` items 61 to 75 are lab runs: queue them with `python -m next.lab run <files>` (see the item), check in once per hour with `tail -n 5`, and report the
result files the item names. They resume from the last saved state after a power cut: run the same command again.

## Mission 6: headers of the Anima files (about 15 minutes, almost no tokens)
The Anima family is written from memory of ComfyUI's code; real files decide whether it is right. In the owner's models folders find the Anima transformer
(file names with `anima` or `cosmos`), the Qwen3 text encoder (`qwen`), the Wan-style VAE (`wan`, `qwen_image_vae`), and, as a control, one SDXL checkpoint. Run
`python scripts/dump_header.py <each file> --out scratch/headers` (reads only the header, a few hundred KB each) and give the owner the `scratch/headers/*.json`
files plus the printed lines. Do not open the JSON in the chat. If a file does not exist, say which kind is missing; do not download models.

## When you are done
List what you could not do and why, in five lines. The owner pastes the reports to the developer; do not summarise results into advice about the code.
