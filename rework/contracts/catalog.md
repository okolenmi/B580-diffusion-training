# Contract: catalog

Implements plans 30 section 13, 41 section 7, 20, 40. Python: `next/graph/catalog.py`.
`GET /catalog` returns:

```jsonc
{
  "catalog_format": 1,
  "nodes": [ NodeInfo ],
  "settings": [ SettingSpec ],            // core + contributed (see settings.md)
  "handle_types": { "ClipLPart": ["ClipLPart", "ModelPart", "object"] },
  "families": [ { "id": "sdxl", "label": "SDXL" } ],
  "load_errors": [ { "module": "...", "error": "..." } ]
}
```

`NodeInfo`:

```jsonc
{
  "class_name": "LoadModelNode",
  "display_name": "Load Model",
  "category": "Model",                    // palette location, "A / B" for nested
  "doc": "...",
  "inputs":  [ PortInfo ],
  "outputs": [ PortInfo ],
  "groups":  [ { "id": "main", "label": "Source", "collapsed": false,
                 "advanced": false, "visible_when": null, "doc": "" } ],
  "events_in":  [ EventPortInfo ],
  "events_out": [ EventPortInfo ],
  "modules": [ DisplayModuleInfo ],
  "reads_settings": [ "device.default" ],
  "settings": [ SettingSpec ],            // contributed by this class
  "dynamic": null | { "depends_on": ["scope", "file"], "debounce_ms": 300 },
  "has_diagnostics": false,
  "accepts_unfinished": false
}
```

`PortInfo`:

```jsonc
{
  "name": "file", "label": "File", "type": "str",
  "type_mro": ["str", "object"],          // names, most specific first; wire check
  "required": true, "default": null, "doc": "",
  "group": "main", "advanced": false, "unit": "",
  "minimum": null, "maximum": null, "step": null,
  "choices": null,                        // closed list for str ports
  "picker": null,                         // scope name: list files from that folder
  "visible_when": null,                   // [other_port, value | [values]]
  "widget_only": false,                   // no wire socket
  "multi": false,                         // several wires allowed; arrives as a list
  "field": true,                          // primitive type: shown as a widget
  "ui": { "widget": "path", "rows": 3, "placeholder": "" }   // advisory hints only
}
```

`EventPortInfo`: `{name, label, doc, payload, payload_mro}`.

`DisplayModuleInfo`: the existing module shape (`name, kind, title, series[],
pattern, exclude, cadence, x, scale, value_key, total_key, reference_key, doc,
size`), unchanged in meaning.

Rules:

1. `catalog_format` increments only on an incompatible change to this document.
2. `type_mro` lets the browser decide wire compatibility exactly as the server does:
   an output satisfies an input when the input's type name is in the output's
   `type_mro`.
3. Order of `inputs` and `groups` is display order.
4. The catalog never instantiates a node and never imports torch.
