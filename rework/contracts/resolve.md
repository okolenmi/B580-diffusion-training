# Contract: shape resolution

Implements plan 41 sections 2, 6, 7 and plan 30 section 6.3. Python:
`next/graph/resolve.py`, hook `Node.resolve(params, env) -> Resolution` (`env` is the `Environment`: named machine services such as the model library; may be absent).

A node whose ports depend on its params declares `DYNAMIC = DynamicShape(depends_on,
debounce_ms)` and overrides `resolve`. All other nodes use the default, which returns
the static ports. `resolve` runs in the **server process**, while a person edits; it is
pure, fast (about 100 ms), imports no torch, reads at most one bounded file header
through the shared cache, and never raises for bad input: it returns an `error` state.

`POST /graphs/nodes/{class_name}/resolve`, body `{ "params": {...}, "settings": {...} }`:

```jsonc
{
  "state": "ready" | "needs_choice" | "error",
  "inputs":  [ PortInfo ],      // full resolved input list (static + dynamic)
  "outputs": [ PortInfo ],
  "choices": { "family": ["sdxl", "anima"] },   // computed dropdowns by port name
  "info": "SDXL checkpoint: denoiser + clip_l + clip_g + vae, bf16, 6.9 GB",
  "message": null,              // human reason for error / needs_choice
  "focus": null,                // port name the editor should highlight
  "detail": {}                  // free JSON for the node's own UI (e.g. adapter groups)
}
```

Client-only state: `resolving` (request in flight). It is never sent by the server.

States:

* `ready`: the shape is final for these params.
* `needs_choice`: ambiguous; `choices` and `focus` say what to decide; outputs may be
  empty. The validator reports it as an **error** (`needs_choice`).
* `error`: could not be resolved; `message` says why; the editor keeps the last good
  shape. The validator reports `resolve_error`.

Rules:

1. Responses are idempotent for equal `(class, params, settings)`.
2. The client tags requests with a sequence number and ignores superseded answers.
3. The graph validator calls the same `resolve` for every node, so what the editor
   showed is what a run is checked against.
4. A wire whose port vanished after a resolve is **dangling**: it stays in the document
   and the validator reports `unknown_output_port` / `unknown_input_port` (plan 41,
   section 6).
