"""Template documents -- a saved *page* built on a graph.

A template is what makes a graph usable by someone who should not have to
meet the graph. It is a board of **blocks**, each pointing at one *module*
(a node's editable field, a node's live view, or one of the graph's own
settings/views -- see ``domain/modules.py``), placed on a column grid, plus
a **snapshot of the graph** those modules belong to. The graph editor stays
the place graphs are authored; the template is a *view* over one.

Two ways to use a template page, one document:

* **constructed** (default) -- the blocks are the interface: the person
  changes the fields that were left exposed, presses Run, and watches the
  live blocks. Nothing about the layout is editable.
* **edit** -- the layout is: blocks move, resize, expand and collapse, and
  anything the graph offers but the page does not show can be added from a
  tray. Hiding a field is just not placing it; it keeps its saved value.

The same document also describes a *monitor window* (``kind="monitor"``):
a board of display blocks only, bound to a running graph. One engine, two
hosts.

This module is the **shape** check and normaliser -- pure, no I/O, no
``nodes`` import. Whether a block's target still *exists* in the graph is a
different question (it needs the node catalog) and is answered by the
resolve use case, because a template whose graph changed under it must still
load, and show the orphaned block as orphaned, rather than be refused
(docs/design/15, "missing targets degrade visibly").

Stored verbatim (like a saved graph), so unknown keys survive a round trip
and a newer frontend's additions are not silently dropped by an older
backend.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .modules import KINDS, SCOPES

TEMPLATE_FORMAT = 1

#: A page that edits and runs a graph, or a window that only watches one.
TEMPLATE_KINDS: tuple[str, ...] = ("template", "monitor")

DEFAULT_COLUMNS = 12
MIN_COLUMNS = 1
MAX_COLUMNS = 24
MAX_BLOCKS = 200
MAX_ROWS = 10_000
MAX_BLOCK_ROWS = 200
MAX_TITLE = 120
MAX_OPTIONS_BYTES = 4096
MAX_NODE_ID = 120

_BLOCK_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_template_doc(doc: Any) -> list[str]:
    """Every shape problem in ``doc`` as ``"path: message"`` strings
    (empty = well-formed). Collects them all rather than stopping at the
    first, for the same reason graph validation does: one round trip should
    tell the editor everything to fix."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["template: must be an object"]

    kind = doc.get("kind", "template")
    if kind not in TEMPLATE_KINDS:
        problems.append(f"kind: {kind!r} is not one of {list(TEMPLATE_KINDS)}")

    columns = doc.get("columns", DEFAULT_COLUMNS)
    if not _is_int(columns) or not MIN_COLUMNS <= columns <= MAX_COLUMNS:
        problems.append(f"columns: must be an integer {MIN_COLUMNS}..{MAX_COLUMNS}, got {columns!r}")
        columns = DEFAULT_COLUMNS

    graph = doc.get("graph")
    if graph is None and kind == "template":
        problems.append("graph: a template needs the graph snapshot it runs")
    if graph is not None:
        if not isinstance(graph, dict):
            problems.append("graph: must be an object")
        else:
            if not isinstance(graph.get("nodes", []), list):
                problems.append("graph.nodes: must be a list")
            if not isinstance(graph.get("edges", []), list):
                problems.append("graph.edges: must be a list")

    source = doc.get("source_graph")
    if source is not None and not isinstance(source, str):
        problems.append("source_graph: must be a string or null")

    blocks = doc.get("blocks", [])
    if not isinstance(blocks, list):
        problems.append("blocks: must be a list")
        return problems
    if len(blocks) > MAX_BLOCKS:
        problems.append(f"blocks: at most {MAX_BLOCKS}, got {len(blocks)}")

    seen: set[str] = set()
    for index, block in enumerate(blocks[:MAX_BLOCKS]):
        where = f"blocks[{index}]"
        if not isinstance(block, dict):
            problems.append(f"{where}: must be an object")
            continue
        block_id = block.get("id")
        if not isinstance(block_id, str) or not _BLOCK_ID.match(block_id):
            problems.append(f"{where}.id: must match [A-Za-z0-9_-]{{1,40}}, got {block_id!r}")
        elif block_id in seen:
            problems.append(f"{where}.id: duplicate id {block_id!r}")
        else:
            seen.add(block_id)

        ref = block.get("ref")
        if not isinstance(ref, dict):
            problems.append(f"{where}.ref: must be an object")
        else:
            scope = ref.get("scope")
            if scope not in SCOPES:
                problems.append(f"{where}.ref.scope: {scope!r} is not one of {list(SCOPES)}")
            if ref.get("kind") not in KINDS:
                problems.append(f"{where}.ref.kind: {ref.get('kind')!r} is not one of {list(KINDS)}")
            name = ref.get("name")
            if not isinstance(name, str) or not name:
                problems.append(f"{where}.ref.name: must be a non-empty string")
            node = ref.get("node")
            if scope == "node":
                if not isinstance(node, str) or not node or len(node) > MAX_NODE_ID:
                    problems.append(f"{where}.ref.node: a node block needs a node id")
            elif scope == "graph" and node is not None:
                problems.append(f"{where}.ref.node: a graph block has no node")

        for key, low in (("x", 0), ("y", 0), ("w", 1), ("h", 1)):
            value = block.get(key)
            if not _is_int(value) or value < low:
                problems.append(f"{where}.{key}: must be an integer >= {low}, got {value!r}")
        if _is_int(block.get("x")) and _is_int(block.get("w")) and block["x"] + block["w"] > columns:
            problems.append(
                f"{where}: x + w = {block['x'] + block['w']} runs past the {columns}-column grid"
            )
        if _is_int(block.get("y")) and block["y"] > MAX_ROWS:
            problems.append(f"{where}.y: at most {MAX_ROWS}")
        if _is_int(block.get("h")) and block["h"] > MAX_BLOCK_ROWS:
            problems.append(f"{where}.h: at most {MAX_BLOCK_ROWS}")

        if "collapsed" in block and not isinstance(block["collapsed"], bool):
            problems.append(f"{where}.collapsed: must be a boolean")
        title = block.get("title")
        if title is not None and (not isinstance(title, str) or len(title) > MAX_TITLE):
            problems.append(f"{where}.title: a string of at most {MAX_TITLE} characters")
        options = block.get("options")
        if options is not None:
            if not isinstance(options, dict):
                problems.append(f"{where}.options: must be an object")
            else:
                try:
                    size = len(json.dumps(options))
                except (TypeError, ValueError):
                    problems.append(f"{where}.options: must be JSON")
                else:
                    if size > MAX_OPTIONS_BYTES:
                        problems.append(f"{where}.options: at most {MAX_OPTIONS_BYTES} bytes, got {size}")
    return problems


def normalize_template_doc(doc: dict) -> dict:
    """The stored shape: format stamped, defaults filled, unknown keys kept.

    Call only on a document ``validate_template_doc`` accepted."""
    out = dict(doc)
    out["format"] = TEMPLATE_FORMAT
    out["kind"] = doc.get("kind", "template")
    out["columns"] = doc.get("columns", DEFAULT_COLUMNS)
    out["source_graph"] = doc.get("source_graph")
    out["blocks"] = [
        {
            **block,
            "collapsed": bool(block.get("collapsed", False)),
            "title": block.get("title"),
            "options": dict(block.get("options") or {}),
        }
        for block in doc.get("blocks", [])
    ]
    return out
