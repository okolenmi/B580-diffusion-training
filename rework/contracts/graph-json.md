# Contract: graph document

Implements plan 60 section 4, plan 10 section 4, plan 20 section 6. No earlier
format is read (plan 90 section 0). Python: `next/graph/document.py`.

```jsonc
{
  "format": 3,
  "settings": { "memory.vram_max_mb": "auto", "device.default": "xpu" },
  "nodes": [
    {
      "uid": "n_k3f9a2",              // identity; assigned at creation, never edited
      "title": "Load model",          // editable display name, free text, may repeat
      "class_name": "LoadModelNode",  // node class
      "params": { "scope": "checkpoints", "file": "x.safetensors" },
      "layout": { "x": 80, "y": 120, "w": 280, "h": null,
                  "collapsed": false, "groups": { "advanced": true } },
      "addons": [                     // optional; omitted when empty (plan 31)
        { "id": "a1", "kind": "RegionLadder", "enabled": true, "params": { "start": 9 } }
      ]
    }
  ],
  "edges": [
    { "from_node": "n_k3f9a2", "from_port": "denoiser",
      "to_node": "n_77ab10",  "to_port": "model" }
  ],
  "event_edges": [
    { "from_node": "n_a", "from_port": "epoch_ending",
      "to_node": "n_b",   "to_port": "epoch_ending" }
  ],
  "views": [                           // plan 60; opaque to graph/ (kept verbatim), see "Views" below
    { "id": "v1", "title": "Monitor", "mode": "monitor", "columns": 12,
      "blocks": [ { "id": "b1", "ref": { "scope": "node", "node": "n_a", "kind": "display", "name": "loss" },
                    "x": 0, "y": 0, "w": 12, "h": 6, "collapsed": false, "title": null, "options": {} } ] }
  ],
  "addons": [                          // optional; addons of the graph itself (event addons only)
    { "id": "g1", "kind": "EveryNSteps", "enabled": true, "params": { "every": 250 } }
  ]
}
```

Rules:

1. `format` is an integer; the reader accepts exactly the current one and rejects
   others with a clear message (no migration).
2. `uid`: `n_` plus six lowercase base32 characters, unique within a graph. Every
   reference between parts of the document uses `uid`. `title` is presentation only.
3. `params` holds values for **widget ports** (fields). A key that is not an input
   port of the node's resolved shape is an `unknown_param` issue. A value on a port
   that also has a wire is ignored in favour of the wire.
4. `edges`: data plane. One edge per ordinary input port; **several** into a port
   declared `multi`, in **list order** (that order is the argument order). `from_port`
   and `to_port` name resolved ports.
5. `event_edges`: event plane. Many-to-many is allowed. Ports named are event ports.
6. `settings`: dotted keys; values validated against the setting specs. Unknown keys are
   kept and reported as warnings.
7. `layout` and `views` are presentation; the runtime ignores them; the server stores
   them verbatim. `layout.groups` maps a group id to its collapsed/expanded override.
8. Unknown top-level keys are preserved on a round trip.
9. A check or a memory preview strips `views` and `layout`; a run is sent the whole document and its stored snapshot keeps both, so the run can be shown as it was drawn (an instance). The runtime and the memory fingerprint ignore them.
10. `addons` (on a node, and at the top level for the graph): ordered; `id` is stable inside its host (`[a-z][a-z0-9_]*`),
    `kind` an addon class name. A disabled addon is neither built nor checked; its params stay. An addon's parameter that is
    wired is the port `<id>.<param>` of its host, and a slot filled by a wire is the port `slot:<slot>`. An event wire may end at
    `<uid>.<id>` or `graph.<id>`. Rules and issue codes (`addon_*`): `rework/31-node-addons.md`.

## Views

`views` is a list of boards that live in the graph document: `{id, title, mode: "page" | "monitor", columns, blocks}`. A block is
`{id, ref, x, y, w, h, collapsed, title, options}` (the board's block, `frontend/js/board/model.js`); `ref` is
`{scope: "node", node: <uid>, kind: "field" | "display", name}` or `{scope: "graph", kind, name}` (a graph setting such as
`memory.vram_max_mb`, or the run status), or `{scope: "note", kind: "note", name}` (free text kept in `options.text`, at most 2000 characters;
no target, so it cannot be orphaned; allowed in both kinds of view). A monitor view holds displays only; a page view also holds fields.

* One truth: a field block edits the node parameter (or setting) of the open graph itself, so the canvas and the page show the
  same value; Run on a page is the editor's Run. No snapshot of the graph is stored in a view.
* Whether a block's target still exists is answered by `POST /templates/resolve` (a draft document whose `graph` is the open
  graph); a block that lost its node is kept and shown as orphaned.
* The runtime ignores views and a run payload strips them. Views are saved with the graph in the library and in the browser's draft.
* Buttons other than Run/Stop (triggering an event input) wait for a control channel into the running process (plan 60 section 8).
