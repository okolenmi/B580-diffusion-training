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
                  "collapsed": false, "groups": { "advanced": true } }
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
  "views": []                          // plan 60; opaque to graph/ (kept verbatim)
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
9. A run payload strips `views` and `layout`.
