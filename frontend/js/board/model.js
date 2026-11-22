/* ---------------------------------------------------------------------------
   board/model.js -- a board's document rules, as pure functions.

   A board is `{columns, blocks, graph}`: a grid of *blocks* (cards) that point at modules of a graph, and the "draft graph"
   whose parameter objects the field blocks edit (an App's snapshot, or a stand-in over the open graph document). The stored
   shapes are backend/domain/board.py (a view) and backend/domain/app.py (an app); a block is

     {id, ref: {scope: "node"|"graph"|"note", node?, kind: "field"|"display"|"note", name}, x, y, w, h, collapsed, title, style, options}

   or, for a section / tabbed card, `container: {columns, tabs: [{id, title, blocks}]}` in place of `ref`. Ids are unique across the whole
   document, containers included, so every id-minting function takes the set of ids in use.

   Nothing in this file touches the DOM, the network or a clock, so every
   rule a person relies on -- "what can I add", "which fields does the
   starter page expose", "what exactly will Run send" -- is checked by
   number in frontend/tests/board_model.test.mjs.

   The *modules* argument is what the backend's POST /boards/modules
   returns: {nodes: [{id, class_name, display_name, known, fields,
   displays, stream}], graph_fields, graph_displays}. The backend, not this
   file, decides what a node offers; this file only arranges it.
   --------------------------------------------------------------------------- */


/* This file re-exports the document rules, which live in one module per concern:
   doc.js (identity, walking, finding), card_kinds.js, tray.js, tree_edit.js, starter.js, values.js, dirty.js. */

export * from "./doc.js";
export * from "./card_kinds.js";
export * from "./tray.js";
export * from "./tree_edit.js";
export * from "./starter.js";
export * from "./values.js";
export * from "./dirty.js";
