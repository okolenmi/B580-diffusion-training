# 06 -- Frontend visual smoke checklist (browser-enabled session)

Status: **written 2026-10-01; not yet executed.** The session that
shipped M6 was console-only (no desktop browser connected), so the
pages were verified by curl (status codes, envelopes, SSE frames) plus
manual JS review -- never actually rendered. Run this checklist from a
session where the browser tools work (OpenCode desktop app with the
experimental browser setting connected), or any other real browser.

## Why this exists

`backend/tests/run_all.py` and the full gate cover the API, SSE frame
contracts, and that the pages/assets *serve*. They cannot cover
JavaScript that actually runs: module wiring, DOM lookups, runtime
exceptions in event handlers. This checklist is that missing layer.

## Prerequisites

* Browser tools connected in the session (desktop app, experimental
  browser setting on), with devtools/console access.
* `/home/okolenmi/comfy/venv/bin/python` (plain `python3` has no torch
  and the backend imports it).

## Setup

```bash
cd /home/okolenmi/Desktop/B580-diffusion-training
rm -f /tmp/opencode/smoke.db
BACKEND_DB_PATH=/tmp/opencode/smoke.db \
  /home/okolenmi/comfy/venv/bin/python -m backend.cli --port 8766
```

A scratch DB keeps the check independent of real training history.

## 1. App shell + training controls -- `http://127.0.0.1:8766/`

* [ ] Page renders: sidebar nav, training controls card, runs table.
* [ ] Console clean (no errors/warnings from our modules).
* [ ] Start-options: with no config path chosen the page shows the
      placeholder (the API deliberately answers 422 `invalid_query`
      for an empty `path` -- that is contract, not a bug); choosing a
      path lists options.
* [ ] `GET /runs/active` 404 (`no_active_run`) renders as "no active
      run", not an exception.
* [ ] Runs history lists rows; opening a run log shows lines.
* [ ] `/api/v1/events` SSE connects (network tab: `stream_opened`).

## 2. Monitor dashboard -- `http://127.0.0.1:8766/monitor/<any-id>`

* [ ] Page renders: chart canvas, series list, controls; console clean.
* [ ] Stream connects (first frame `{"type":"connected"}`).
* [ ] Empty-id history: absent series show gaps/em dashes -- **never
      fabricated zeros** (hard rule, docs 03 section 4).
* [ ] Optional, needs a live producer: run something that reports to
      the monitor id -> points appear live; reload mid-stream ->
      history replays (chart restores without a rerun); a `clear`
      frame empties the chart.
* [ ] CSV export and series toggles work on real data.

## 3. Graph editor -- `http://127.0.0.1:8766/graph` (M7+)

* [ ] Palette lists the catalog grouped by domain; unknown/failed
      modules surface as `load_errors`, not a blank page.
* [ ] Add a node, move it, connect output->input, delete; canvas keeps
      positions after a reload round-trip (library save/load).
* [ ] Params form: typed widgets (choices, defaults, paths) edit the
      node's `params`.
* [ ] Validate: clean graph -> ok; broken graph (dup id, missing
      required input) -> complete issue list, localized to nodes.
* [ ] Run: execution appears, per-node progress/status renders, stop
      works, failures show per-node errors (status `error`, not
      `finished`).
* [ ] Library: save, overwrite, load, delete; importing a legacy
      `ng_graph_v1` entry from localStorage produces a loadable graph.
* [ ] Old-server port conflict check is *not* needed here: the editor
      talks only to `/api/v1/graphs/*` (verify in network tab).

## 4. Record

Paste screenshots of each page into the session and list pass/fail
per checkbox. Fix regressions in the milestone that owns the code --
do not adjust the checklist to match broken behavior.

## Known deferred (not bugs)

* Previews, per-run `/events` history, `logs/clear`, `/files/{kind}`
  -- deferred per docs 03 section 3.1.
* Graph autosave to localStorage: the new frontend never writes
  `ng_graph_v1`; the key is read once as an import source (docs 03
  section 5).
* Training-loop behavioral tests stay deferred until the full server
  rework is done (user decision).
