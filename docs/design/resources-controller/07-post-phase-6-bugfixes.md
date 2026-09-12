*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

## Post-Phase-6 bug fixes, from real usage

Four real bugs, found by actually using the editor rather than by
review, fixed together:

- `server/static/nodegraph.js`'s `fetchDiagnostics()` called
  `/nodegraph/node/{class_name}/diagnostics` -- missing the `/api`
  prefix every other endpoint in this file already uses
  (`server/main.py` mounts `nodegraph_router` under `/api`). A plain
  URL typo, 404ing the live-diagnostics call from Phase 5 on every
  request. Fixed to `/api/nodegraph/node/{class_name}/diagnostics`.
- Toggling a `Port.visible_when`-gated checkbox, or a live diagnostics
  response changing what text is shown, both change a node's own
  height -- and neither `updateFieldVisibility()` nor
  `fetchDiagnostics()` told the canvas to re-measure that node's port
  positions or redraw its wires afterward. Visible result: a wire into
  or out of the node stayed drawn at its pre-toggle position until
  some unrelated action (moving the node) happened to trigger a fresh
  measurement. Fixed: both now call a new `remeasureAndRedraw(node)`
  (`measurePorts()` + `redrawWires()`) after mutating the DOM.
- `LoRATrainingConfigNode`'s `alpha` defaulted to `1.0` against a
  `rank` default of `64` -- under `ClassicLoRAScaling`
  (`nodes/model/lora_scaling.py`, this project's own current default
  policy), the actual multiplier applied is `alpha/rank`, so that
  combination gave a scale of `1/64 ≈ 0.016`: a very weak LoRA by the
  standards of the wider ecosystem, and not a default a person new to
  what alpha even is should be expected to already know to correct.
  Changed `alpha`'s default to `64.0`, matching `rank`'s own default
  -- scale `1.0` out of the box, the common "alpha equal to rank"
  starting point -- and documented the actual formula and the
  rank/alpha interaction directly in both Ports' own `doc` text.
- Several `Port.doc` strings (this document's own file path, "Phase
  N," "see this class's own docstring") had leaked from planning notes
  into what the graph editor shows as a Port's hover tooltip -- a real
  user reads that text with no access to this document or this
  codebase's own source, so a reference meaningless outside either is
  just confusing there. Every `Port.doc` across
  `resources_controller.py`/`lora_training_config.py` audited (an AST
  walk over every `Port(...)` call's own `doc=` value, not a
  string-literal guess) and rewritten to stand on its own.
