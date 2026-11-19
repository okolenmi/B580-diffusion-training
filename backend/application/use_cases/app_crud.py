"""Save / get / list / delete / diff apps.

Small use cases in one file because they are one thing: the library of apps. The rules are the graph library's (trimmed 1..120-char
names, upsert answering 201 vs 200, 404 for an unknown name) plus the shape check in ``domain/app.py``, which runs on save so a document
the frontend cannot render never reaches the table -- while whether a card's *target* still exists is deliberately not checked here
(an app whose snapshot changed must still be loadable; see ``ResolveBoard``).
"""

from __future__ import annotations

from ...domain.app import diff_graphs, graph_hash, normalize_app_doc, validate_app_doc
from ..dto import (
    AppDiffResult,
    DeleteAppResult,
    SavedAppDTO,
    SavedAppListResult,
    SaveAppResult,
    to_saved_app_dto,
    to_saved_app_summary,
)
from ..errors import AppInvalidError, AppNotFoundError, InvalidQueryError
from ..limits import MAX_APP_DESCRIPTION
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


class DiffApp:
    """What "update from graph" would do: the changes by node, and which cards would lose their target against the new graph."""

    def __init__(self, library: AppLibrary, resolve: ResolveBoard) -> None:
        self._library = library
        self._resolve = resolve

    def execute(self, name: str, graph: dict) -> AppDiffResult:
        clean = _clean_name(name)
        saved = self._library.get(clean)
        if saved is None:
            raise AppNotFoundError(f"no saved app named {clean!r}")
        new = normalize_graph_payload(graph)
        difference = diff_graphs(saved.doc["graph"], new)
        resolution = self._resolve.execute(saved.doc, graph=new)
        return AppDiffResult(
            added=tuple(difference["added"]), removed=tuple(difference["removed"]), changed=tuple(difference["changed"]),
            wiring_changed=difference["wiring_changed"], settings_changed=difference["settings_changed"],
            orphaned_cards=tuple(b["id"] for b in resolution.blocks if b["status"] not in (STATUS_OK, STATUS_WIRED)),
            new_version=graph_hash(new) != saved.graph_hash,
        )
