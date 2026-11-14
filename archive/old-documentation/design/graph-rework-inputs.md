# Graph rework inputs (READ-ONLY investigation, 2026-10-10)

Each item marked *verified* (measured or read in code here) or
*reported by maintainer, unverified*. No code changed for this document.

## 1. The memory budget does not take effect on graph runs — verified in code

`backend/infrastructure/graph/` (catalog, discovery, introspect, runtime)
contains zero references to `resource_control`, budget, or
`VRAMBudgetControllerNode` (grep, 2026-10-10). The trainer nodes accept a
`resource_control` port and the managed route enforces through it, but
nothing on the graph path constructs or wires one -- so whatever budget
number exists anywhere else cannot reach a graph run. This is not a
tuning problem; there is no wire.

Claimed vs reality: `docs/design/resources-controller/` describes
controller enforcement as the route's central mechanism (05-phase-5,
09-trainer-integration-and-vram-safety). True for hand-built trainer
nodes with the port wired (hw_validate wires it). Not true for graphs
built in the editor: the port exists on the node, nothing connects it.

Repro: build any training graph in the editor, set any budget, run;
reserved VRAM ignores it. (Code fact needs no run: no wire, no effect.)

## 2. Graph initialisation memory — partly measured, partly pending

*Verified (server side):* node discovery (`NodeRegistry.scan_nodes`)
costs **1.4 s, 682 MB peak host RSS, 35 classes, 0 errors** on this
machine -- cheap. The server is not where init memory goes.

*Verified (model build, same code a graph run executes):* UNet+LoRA
construction peaks at **5,262 MB reserved** (hw floor stages) and takes
~5 s warm after the meta-device change (was ~15-19 s cold, mostly
page-cache + discarded random init). CLIP adds ~1.5 GB on top when it
loads. A graph-run child process pays this on top of the weights already
resident -- that is the init number to beat, and it is a property of the
build, not of the graph layer.

*Reported by maintainer, unverified:* graph initialisation "wants more
memory than is available so it will not start". The numbers above bound
the model-build half; what a full graph *run* holds at start (child
process + model + optimizer + first batch + capture pools if enabled)
was not measured here -- needs a quiet card (B arms hold it through the
study) via `MemProbe` floor stages on a real graph run.

## 3. No graph-settings section — verified in code

`backend/infrastructure/config_ui_data.py` contains zero keys with
"graph" in them (grep). There is nowhere in the settings UI for graph
defaults (budget, capture flags, batching) to live. Related: config
`lr_schedule` / LoRA target keys parse and validate but reach no
schedule/injector node (open.md entry 2026-10-10) -- the settings exist
as data, not as controls, in both places.

## UI/graph usability inventory (code-verified unless noted)

- `frontend/js/views/` has config, datasets, install -- **no run view**.
  There is no view that could show a run (matches the retired-plan
  references to a never-built `views/run.js`; the live nonfinite reader
  is `frontend/js/lib/value.js`). *Verified in code.*
- The 09-event-contract gap (no replay; sleeping tabs believe runs are
  still running) stands as written. *Verified as documented; browser
  behavior reported by maintainer, unverified here.*
- Budget display with no enforcement path (item 1) is the highest-risk
  UI fact: a number the user can set that nothing reads. *Verified.*
