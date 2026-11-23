"""Boards -- cards on a column grid, flat or nested.

A board is what the graph editor's *views* and the *Apps* share: a grid of **cards**, each pointing at one *module* (a node's editable
field, a node's live display, a graph setting or a note; ``domain/modules.py``) or holding a **container** of its own. The frontend's
board engine draws both (``frontend/js/board``).

* A flat board is ``{mode, columns, blocks}``: a view inside a graph document.
* A container is ``{columns, tabs: [{id, title, blocks}]}``: its own grid per tab. A card that carries ``container`` instead of ``ref``
  is a titled section (one tab) or a tabbed card (several); the App's page is the root container.

This module is the **shape** check and the walker over cards -- pure, no I/O. Whether a card's *target* still exists needs the node
catalog and is answered by the resolve use case, because a board whose graph changed must still load and show the orphan as orphaned.

Boards are stored verbatim (unknown keys survive a round trip), so a newer frontend's additions are not dropped by an older backend.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, TypeGuard

from .modules import KINDS, SCOPES

BOARD_MODES: tuple[str, ...] = ("page", "monitor")
CARD_STYLES: tuple[str, ...] = ("plain", "outlined", "accent", "bare")

DEFAULT_COLUMNS = 12
MIN_COLUMNS = 1
MAX_COLUMNS = 24
#: Cards in one document, containers included, at any depth.
MAX_BLOCKS = 200
#: Container cards inside container cards.
MAX_DEPTH = 3
MAX_TABS = 12
MAX_ROWS = 20_000
MAX_BLOCK_ROWS = 400
MAX_TITLE = 120
MAX_OPTIONS_BYTES = 4096
MAX_NODE_ID = 120

_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


class _Walk:
    """What one validation shares: the problems found, the card ids seen (unique across the whole document) and the card count."""

    __slots__ = ("problems", "seen", "cards")

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        self.seen: set[str] = set()
        self.cards = 0


def validate_board_doc(doc: Any) -> list[str]:
    """Every shape problem of a flat board as ``"path: message"`` (empty = well-formed). All are collected, so one round trip tells the
    editor everything to fix."""
    if not isinstance(doc, dict):
        return ["board: must be an object"]
    problems: list[str] = []
    mode = doc.get("mode", "page")
    if mode not in BOARD_MODES:
        problems.append(f"mode: {mode!r} is not one of {list(BOARD_MODES)}")
    columns = _columns(doc.get("columns", DEFAULT_COLUMNS), "columns", problems)
    blocks = doc.get("blocks", [])
    if not isinstance(blocks, list):
        problems.append("blocks: must be a list")
        return problems
    validate_cards(blocks, "blocks", columns, 0, False, _Walk(problems))
    return problems


def validate_container(container: Any, where: str, problems: list[str]) -> None:
    """The shape of a container (an App's page) with a fresh card budget: used for a document's root."""
    _container(container, where, 0, _Walk(problems))


def _columns(value: Any, where: str, problems: list[str]) -> int:
    if not _is_int(value) or not MIN_COLUMNS <= value <= MAX_COLUMNS:
        problems.append(f"{where}: must be an integer {MIN_COLUMNS}..{MAX_COLUMNS}, got {value!r}")
        return DEFAULT_COLUMNS
    return value


def _container(container: Any, where: str, depth: int, walk: _Walk) -> None:
    if not isinstance(container, dict):
        walk.problems.append(f"{where}: must be an object")
        return
    columns = _columns(container.get("columns", DEFAULT_COLUMNS), f"{where}.columns", walk.problems)
    tabs = container.get("tabs")
    if not isinstance(tabs, list) or not 1 <= len(tabs) <= MAX_TABS:
        walk.problems.append(f"{where}.tabs: a list of 1..{MAX_TABS} tabs")
        return
    tab_ids: set[str] = set()
    for index, tab in enumerate(tabs):
        here = f"{where}.tabs[{index}]"
        if not isinstance(tab, dict):
            walk.problems.append(f"{here}: must be an object")
            continue
        tab_id = tab.get("id")
        if not isinstance(tab_id, str) or not _ID.match(tab_id):
            walk.problems.append(f"{here}.id: must match [A-Za-z0-9_-]{{1,40}}, got {tab_id!r}")
        elif tab_id in tab_ids:
            walk.problems.append(f"{here}.id: duplicate id {tab_id!r}")
        else:
            tab_ids.add(tab_id)
        title = tab.get("title")
        if not isinstance(title, str) or not title.strip() or len(title) > MAX_TITLE:
            walk.problems.append(f"{here}.title: a non-empty string of at most {MAX_TITLE} characters")
        cards = tab.get("blocks", [])
        if not isinstance(cards, list):
            walk.problems.append(f"{here}.blocks: must be a list")
            continue
        validate_cards(cards, f"{here}.blocks", columns, depth, True, walk)


def validate_cards(cards: list, where: str, columns: int, depth: int, allow_containers: bool, walk: _Walk) -> None:
    walk.cards += len(cards)
    if walk.cards > MAX_BLOCKS:
        if walk.cards - len(cards) <= MAX_BLOCKS:  # said once, by the list that crosses the line
            walk.problems.append(f"{where}: at most {MAX_BLOCKS} cards in a document, containers included")
        return
    for index, card in enumerate(cards):
        _card(card, f"{where}[{index}]", columns, depth, allow_containers, walk)


def _card(card: Any, where: str, columns: int, depth: int, allow_containers: bool, walk: _Walk) -> None:
    if not isinstance(card, dict):
        walk.problems.append(f"{where}: must be an object")
        return
    card_id = card.get("id")
    if not isinstance(card_id, str) or not _ID.match(card_id):
        walk.problems.append(f"{where}.id: must match [A-Za-z0-9_-]{{1,40}}, got {card_id!r}")
    elif card_id in walk.seen:
        walk.problems.append(f"{where}.id: duplicate id {card_id!r}")
    else:
        walk.seen.add(card_id)

    if "container" in card:
        if "ref" in card:
            walk.problems.append(f"{where}: a container card has no ref")
        if not allow_containers:
            walk.problems.append(f"{where}.container: not allowed in a flat board")
        elif depth >= MAX_DEPTH:
            walk.problems.append(f"{where}.container: containers nest at most {MAX_DEPTH} deep")
        else:
            _container(card["container"], f"{where}.container", depth + 1, walk)
    else:
        _ref(card, where, walk.problems)
    _geometry(card, where, columns, walk.problems)
    _presentation(card, where, walk.problems)


def _ref(card: dict, where: str, problems: list[str]) -> None:
    """The `ref` a card points at: scope/kind/name, and the node-id rules."""
    ref = card.get("ref")
    if not isinstance(ref, dict):
        problems.append(f"{where}.ref: must be an object")
        return
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
    elif scope == "note" and node is not None:
        problems.append(f"{where}.ref.node: a note has no node")
    if ref.get("kind") == "action" and scope != "node":
        problems.append(f"{where}.ref: an action belongs to a node (scope \"node\")")
    if (scope == "note") != (ref.get("kind") == "note"):
        problems.append(f"{where}.ref: scope and kind are both \"note\" for a note, and only then")


def _geometry(card: dict, where: str, columns: int, problems: list[str]) -> None:
    """Where the card sits: x/y/w/h, the column grid, and the row caps."""
    for key, low in (("x", 0), ("y", 0), ("w", 1), ("h", 1)):
        value = card.get(key)
        if not _is_int(value) or value < low:
            problems.append(f"{where}.{key}: must be an integer >= {low}, got {value!r}")
    if _is_int(card.get("x")) and _is_int(card.get("w")) and card["x"] + card["w"] > columns:
        problems.append(f"{where}: x + w = {card['x'] + card['w']} runs past the {columns}-column grid")
    if _is_int(card.get("y")) and card["y"] > MAX_ROWS:
        problems.append(f"{where}.y: at most {MAX_ROWS}")
    if _is_int(card.get("h")) and card["h"] > MAX_BLOCK_ROWS:
        problems.append(f"{where}.h: at most {MAX_BLOCK_ROWS}")


def _presentation(card: dict, where: str, problems: list[str]) -> None:
    """The cosmetic fields: collapsed, title, style and the options blob."""
    if "collapsed" in card and not isinstance(card["collapsed"], bool):
        problems.append(f"{where}.collapsed: must be a boolean")
    title = card.get("title")
    if title is not None and (not isinstance(title, str) or len(title) > MAX_TITLE):
        problems.append(f"{where}.title: a string of at most {MAX_TITLE} characters")
    style = card.get("style")
    if style is not None and style not in CARD_STYLES:
        problems.append(f"{where}.style: {style!r} is not one of {list(CARD_STYLES)}")
    options = card.get("options")
    if options is None:
        return
    if not isinstance(options, dict):
        problems.append(f"{where}.options: must be an object")
        return
    try:
        size = len(json.dumps(options))
    except (TypeError, ValueError):
        problems.append(f"{where}.options: must be JSON")
        return
    if size > MAX_OPTIONS_BYTES:
        problems.append(f"{where}.options: at most {MAX_OPTIONS_BYTES} bytes, got {size}")


def iter_cards(doc: dict) -> Iterator[dict]:
    """Every card that has a ``ref``, at any depth, of a flat board (``blocks``) or an App (``root``). Call on a validated document."""
    yield from _cards_of(doc.get("blocks") or [])
    root = doc.get("root")
    if isinstance(root, dict):
        yield from _cards_of_container(root)


def _cards_of_container(container: dict) -> Iterator[dict]:
    for tab in container.get("tabs", []):
        yield from _cards_of(tab.get("blocks", []))


def _cards_of(cards: list) -> Iterator[dict]:
    for card in cards:
        if "container" in card:
            yield from _cards_of_container(card["container"])
        else:
            yield card


def normalize_container(container: dict) -> dict:
    """A container with defaults filled and every card normalised; unknown keys kept. Call on a validated one."""
    return {
        **container,
        "columns": container.get("columns", DEFAULT_COLUMNS),
        "tabs": [{**tab, "blocks": normalize_cards(tab.get("blocks", []))} for tab in container["tabs"]],
    }


def normalize_cards(cards: list) -> list[dict]:
    out = []
    for card in cards:
        normal = {**card, "collapsed": bool(card.get("collapsed", False)), "title": card.get("title"),
                  "options": dict(card.get("options") or {})}
        if "container" in card:
            normal["container"] = normalize_container(card["container"])
        out.append(normal)
    return out
