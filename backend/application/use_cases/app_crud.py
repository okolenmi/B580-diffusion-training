"""Save / get / list / delete apps, and replace the graph an app owns.

Small use cases in one file because they are one thing: the library of apps. The rules are the graph library's (trimmed 1..120-char
names, upsert answering 201 vs 200, 404 for an unknown name) plus the shape check in ``domain/app.py``, which runs on save so a document
the frontend cannot render never reaches the table -- while whether a card's *target* still exists is deliberately not checked here
(an app whose snapshot changed must still be loadable; see ``ResolveBoard``).
"""

from __future__ import annotations

import json

from ...domain.app import graph_hash, normalize_app_doc, validate_app_doc
from ..dto import (
    DeleteAppResult,
    SavedAppDTO,
    SavedAppListResult,
    SaveAppGraphResult,
    SaveAppResult,
    to_saved_app_dto,
    to_saved_app_summary,
)
from ..errors import AppInvalidError, AppNotFoundError, InvalidQueryError
from ..limits import MAX_APP_DESCRIPTION, MAX_APP_VALUES
from ..ports.app_library import AppLibrary, normalize_app_name
from .graph_modules import STATUS_OK, STATUS_WIRED, ResolveBoard
from .save_graph import normalize_graph_payload


def _clean_name(name: str) -> str:
    try:
        return normalize_app_name(name)
    except ValueError as exc:
        raise InvalidQueryError(str(exc)) from exc


class SaveApp:
    def __init__(self, library: AppLibrary) -> None:
        self._library = library

    def execute(self, name: str, doc: dict, *, description: str = "") -> SaveAppResult:
        clean = _clean_name(name)
        if len(description) > MAX_APP_DESCRIPTION:
            raise InvalidQueryError(f"description must be at most {MAX_APP_DESCRIPTION} characters")
        problems = validate_app_doc(doc)
        if problems:
            raise AppInvalidError(f"app has {len(problems)} problem(s)", details=problems)
        stored = normalize_app_doc(doc)
        # The snapshot obeys the library's own rules (format stamp, memory block checked by value): one definition of a storable graph.
        stored["graph"] = normalize_graph_payload(stored["graph"])
        saved, created = self._library.save(clean, stored, graph_hash=graph_hash(stored["graph"]), description=description)
        return SaveAppResult(app=to_saved_app_dto(saved), created=created)


class SaveAppValues:
    """Keep what a person typed into an app's page, so that opening the page again shows it. Using an app is not publishing it: no version, no new document."""

    def __init__(self, library: AppLibrary) -> None:
        self._library = library

    def execute(self, name: str, values: dict) -> None:
        clean = _clean_name(name)
        unknown = [key for key in values if key not in ("nodes", "memory")]
        if unknown or any(not isinstance(values.get(key, {}), dict) for key in ("nodes", "memory")):
            raise InvalidQueryError("values must be {\"nodes\": {...}, \"memory\": {...}}")
        if len(json.dumps(values)) > MAX_APP_VALUES:
            raise InvalidQueryError(f"values must be at most {MAX_APP_VALUES} bytes")
        if not self._library.put_values(clean, values):
            raise AppNotFoundError(f"no saved app named {clean!r}")


class GetApp:
    def __init__(self, library: AppLibrary) -> None:
        self._library = library

    def execute(self, name: str) -> SavedAppDTO:
        clean = _clean_name(name)
        saved = self._library.get(clean)
        if saved is None:
            raise AppNotFoundError(f"no saved app named {clean!r}")
        return to_saved_app_dto(saved)


class ListApps:
    def __init__(self, library: AppLibrary) -> None:
        self._library = library

    def execute(self) -> SavedAppListResult:
        rows = tuple(to_saved_app_summary(saved) for saved in self._library.list_apps())
        return SavedAppListResult(apps=rows, count=len(rows))


class DeleteApp:
    def __init__(self, library: AppLibrary) -> None:
        self._library = library

    def execute(self, name: str) -> DeleteAppResult:
        clean = _clean_name(name)
        if not self._library.delete(clean):
            raise AppNotFoundError(f"no saved app named {clean!r}")
        return DeleteAppResult(deleted=True)


class SaveAppGraph:
    """Replace the graph an app owns: the graph editor's Save in app mode.

    The graph is the app's own (copied in when the app was made, never linked to the library), so the page, style and description stay
    as they are; what changes is what the page's cards can still reach, which the answer lists (``orphaned_cards``) so the editor can say
    so. A changed run (``graph_hash``) makes the next version, as for any other save.
    """

    def __init__(self, library: AppLibrary, resolve: ResolveBoard) -> None:
        self._library = library
        self._resolve = resolve

    def execute(self, name: str, graph: dict) -> SaveAppGraphResult:
        clean = _clean_name(name)
        saved = self._library.get(clean)
        if saved is None:
            raise AppNotFoundError(f"no saved app named {clean!r}")
        new = normalize_graph_payload(graph)
        doc = {**saved.doc, "graph": new}
        stored, _ = self._library.save(clean, doc, graph_hash=graph_hash(new), description=saved.description)
        resolution = self._resolve.execute(doc, graph=new)
        return SaveAppGraphResult(
            app=to_saved_app_dto(stored),
            orphaned_cards=tuple(b["id"] for b in resolution.blocks if b["status"] not in (STATUS_OK, STATUS_WIRED)),
        )
