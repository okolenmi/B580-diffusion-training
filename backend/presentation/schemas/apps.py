"""Apps: documents, their graph, runs, modules and resolution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...domain.graph import GraphDefinition
from .memory import MemoryOverridesIn


class AppIn(BaseModel):
    """``PUT /apps/{name}``: the app document plus a description.

    ``doc`` is deliberately an open object here: its shape is checked by ``domain/app.validate_app_doc`` (which reports *every* problem
    with a path), not by a pydantic model that would stop at the first and could not carry the keys a newer frontend adds.
    """

    description: str = ""
    doc: dict[str, Any]


class SavedAppOut(BaseModel):
    name: str
    description: str = ""
    version: int
    graph_hash: str
    doc: dict[str, Any]
    card_count: int
    node_count: int
    created_at: datetime
    updated_at: datetime
    #: What was set on the page since the last save, in the shape a run takes; the page puts it over the document's values.
    values: dict[str, Any] = Field(default_factory=dict)


class SavedAppSummaryOut(BaseModel):
    name: str
    description: str = ""
    version: int
    card_count: int
    node_count: int
    created_at: datetime
    updated_at: datetime


class SavedAppListOut(BaseModel):
    apps: list[SavedAppSummaryOut]
    count: int


class DeleteAppOut(BaseModel):
    deleted: bool


class AppValuesIn(BaseModel):
    """``PUT /apps/{name}/values``: the page's current values, in the shape of a run's ``values``."""

    values: dict[str, Any]


class AppGraphIn(BaseModel):
    """``PUT /apps/{name}/graph``: the graph the app owns from now on (the editor's document, layout included)."""

    graph: dict[str, Any]


class AppGraphOut(BaseModel):
    """The saved app, and the cards whose target the new graph no longer has."""

    app: SavedAppOut
    orphaned_cards: list[str]


class AppRunIn(BaseModel):
    """``POST /apps/{name}/run``.

    ``values`` is ``{"nodes": {node_id: {port: value}}, "memory": {key: value}}`` and may only name fields the app exposes."""

    values: dict[str, Any] = Field(default_factory=dict)
    memory_overrides: MemoryOverridesIn | None = None
    #: As for ``POST /graphs/run``: ``queue`` waits behind whatever lives or waits (the page's "Add to queue").
    mode: Literal["auto", "parallel", "queue"] = "auto"


class ModulesIn(BaseModel):
    """``POST /boards/modules``: a graph document (the editor's payload shape)."""

    graph: dict[str, Any]

    def to_definition(self) -> GraphDefinition:
        return GraphDefinition.from_dict(self.graph)


class GraphModulesOut(BaseModel):
    """Everything a page could bind to. The per-node and per-module dicts
    are open on purpose -- they mirror ``nodes/ui.py`` descriptors, which
    grow without a schema release."""

    nodes: list[dict[str, Any]]
    graph_fields: list[dict[str, Any]]
    graph_displays: list[dict[str, Any]]


class ResolveIn(BaseModel):
    """``POST /boards/resolve``: a draft board (a view, or an app's document), optionally with the graph to resolve it against."""

    doc: dict[str, Any]
    graph: dict[str, Any] | None = None


class BlockStatusOut(BaseModel):
    id: str
    status: str
    detail: str = ""


class ResolveOut(BaseModel):
    modules: GraphModulesOut
    blocks: list[BlockStatusOut]
    issues: list[dict[str, Any]]


def saved_app_out(dto) -> SavedAppOut:
    return SavedAppOut(
        name=dto.name, description=dto.description, version=dto.version, graph_hash=dto.graph_hash, doc=dto.doc,
        card_count=dto.card_count, node_count=dto.node_count, created_at=dto.created_at, updated_at=dto.updated_at, values=dto.values,
    )


def saved_app_list_out(result) -> SavedAppListOut:
    return SavedAppListOut(
        apps=[
            SavedAppSummaryOut(
                name=a.name, description=a.description, version=a.version, card_count=a.card_count,
                node_count=a.node_count, created_at=a.created_at, updated_at=a.updated_at,
            )
            for a in result.apps
        ],
        count=result.count,
    )


def graph_modules_out(modules) -> GraphModulesOut:
    return GraphModulesOut(
        nodes=[dict(n) for n in modules.nodes],
        graph_fields=[dict(f) for f in modules.graph_fields],
        graph_displays=[dict(d) for d in modules.graph_displays],
    )


def resolve_out(resolution) -> ResolveOut:
    return ResolveOut(
        modules=graph_modules_out(resolution.modules),
        blocks=[BlockStatusOut(**b) for b in resolution.blocks],
        issues=[dict(i) for i in resolution.issues],
    )


__all__ = ["AppIn", "SavedAppOut", "SavedAppSummaryOut", "SavedAppListOut", "DeleteAppOut", "AppGraphIn", "AppGraphOut", "AppRunIn", "AppValuesIn", "ModulesIn", "GraphModulesOut", "ResolveIn", "BlockStatusOut", "ResolveOut", "saved_app_out", "saved_app_list_out", "graph_modules_out", "resolve_out"]
