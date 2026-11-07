"""The one place where the backend names the new node tree.

``next`` is the working name of the rewrite (rework/90-build-plan.md); the swap in phase 9 moves its
contents to the repository root, and this module is then the only import that changes.
"""

from __future__ import annotations

from next.families.api.library import ModelLibrary
from next.families.registry import list_families
from next.graph.catalog import build_catalog
from next.graph.node import Environment
from next.graph.registry import NodeRegistry
from next.graph.resolve import resolve_node

__all__ = ["Environment", "ModelLibrary", "NodeRegistry", "build_catalog", "list_families", "resolve_node"]
