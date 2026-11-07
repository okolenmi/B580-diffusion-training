"""ContractNodeCatalog -- the NodeCatalog port over ``next/graph``.

The catalog document is built once and kept until a refresh: it is static for a given set of node
packages. A resolve is answered per call, with the machine's model folders as the environment so that
a node that reads a file header (Load Model) sees the same files the run will.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from ...application.errors import NodeClassNotFoundError
from ...application.ports.node_catalog import NodeCatalog
from . import tree

MODEL_SCOPES = ("diffusion_models", "text_encoders", "vae")


def model_folders(checkpoints: Path, loras: Path) -> dict[str, list[Path]]:
    """ComfyUI's layout: the other model folders sit beside ``checkpoints``."""
    folders = {"checkpoints": [checkpoints], "loras": [loras]}
    for scope in MODEL_SCOPES:
        if (checkpoints.parent / scope).is_dir():
            folders[scope] = [checkpoints.parent / scope]
    return folders


class ContractNodeCatalog(NodeCatalog):
    def __init__(self, registry: object | None = None,
                 environment: Callable[[], tree.Environment] | None = None) -> None:
        self._registry = registry or tree.NodeRegistry.default()
        self._environment = environment or tree.Environment
        self._lock = threading.Lock()
        self._document: dict | None = None

    def catalog(self, *, refresh: bool = False) -> dict:
        with self._lock:
            if refresh:
                self._registry.reload()
                self._document = None
            if self._document is None:
                self._document = tree.build_catalog(self._registry, families=tree.list_families())
            return self._document

    def resolve(self, class_name: str, params: dict, settings: dict) -> dict:  # noqa: ARG002 -- the contract carries settings; a shape does not read them yet
        classes, _ = self._registry.load()
        if class_name not in classes:
            raise NodeClassNotFoundError(f"unknown node class {class_name!r}")
        return tree.resolve_node(self._registry, class_name, params, self._environment())
