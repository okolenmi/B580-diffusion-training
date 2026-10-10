"""Save / get / list / delete templates.

Four small use cases in one file because they are one thing: the library
of pages. The rules are the graph library's (trimmed 1..120-char names,
upsert answering 201 vs 200, 404 for an unknown name) plus the shape
check in ``domain/template.py``, which runs on save so a document the
frontend cannot render never reaches the table -- while whether a block's
*target* still exists is deliberately not checked here (a page whose graph
changed must still be loadable; see ``ResolveTemplate``).
"""

from __future__ import annotations

from ..dto import (
    DeleteTemplateResult,
    SavedTemplateDTO,
    SavedTemplateListResult,
    SaveTemplateResult,
    to_saved_template_dto,
    to_saved_template_summary,
)
from ..errors import InvalidQueryError, TemplateInvalidError, TemplateNotFoundError
from ..limits import MAX_TEMPLATE_DESCRIPTION
from ..ports.template_library import TemplateLibrary, normalize_template_name
from ...domain.template import normalize_template_doc, validate_template_doc
from .save_graph import normalize_graph_payload


def _clean_name(name: str) -> str:
    try:
        return normalize_template_name(name)
    except ValueError as exc:
        raise InvalidQueryError(str(exc)) from exc


class SaveTemplate:
    def __init__(self, library: TemplateLibrary) -> None:
        self._library = library

    def execute(
        self, name: str, doc: dict, *, description: str = ""
    ) -> SaveTemplateResult:
        clean = _clean_name(name)
        if len(description) > MAX_TEMPLATE_DESCRIPTION:
            raise InvalidQueryError(
                f"description must be at most {MAX_TEMPLATE_DESCRIPTION} characters"
            )
        problems = validate_template_doc(doc)
        if problems:
            raise TemplateInvalidError(
                f"template has {len(problems)} problem(s)", details=problems
            )
        stored = normalize_template_doc(doc)
        if stored.get("graph") is not None:
            # The snapshot obeys the library's own rules (format stamp,
            # memory block checked by value) -- one definition of a
            # storable graph, not two.
            stored["graph"] = normalize_graph_payload(stored["graph"])
        saved, created = self._library.save(
            clean, stored, kind=stored["kind"], description=description
        )
        return SaveTemplateResult(template=to_saved_template_dto(saved), created=created)


class GetTemplate:
    def __init__(self, library: TemplateLibrary) -> None:
        self._library = library

    def execute(self, name: str) -> SavedTemplateDTO:
        clean = _clean_name(name)
        saved = self._library.get(clean)
        if saved is None:
            raise TemplateNotFoundError(f"no saved template named {clean!r}")
        return to_saved_template_dto(saved)


class ListTemplates:
    def __init__(self, library: TemplateLibrary) -> None:
        self._library = library

    def execute(self, kind: str | None = None) -> SavedTemplateListResult:
        if kind is not None and kind not in ("template", "monitor"):
            raise InvalidQueryError(f"kind must be 'template' or 'monitor', got {kind!r}")
        rows = tuple(
            to_saved_template_summary(saved)
            for saved in self._library.list_templates(kind)
        )
        return SavedTemplateListResult(templates=rows, count=len(rows))


class DeleteTemplate:
    def __init__(self, library: TemplateLibrary) -> None:
        self._library = library

    def execute(self, name: str) -> DeleteTemplateResult:
        clean = _clean_name(name)
        if not self._library.delete(clean):
            raise TemplateNotFoundError(f"no saved template named {clean!r}")
        return DeleteTemplateResult(deleted=True)
