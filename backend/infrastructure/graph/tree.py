"""The one place where the backend names the new node tree.

``next`` is the working name of the rewrite (rework/90-build-plan.md); the swap in phase 9 moves its
contents to the repository root, and this module is then the only import that changes.
"""

from __future__ import annotations

from next.data.library import report as library_report
from next.data.store import DatasetStore
from next.families.api.library import ModelLibrary, comfy_folders
from next.families.registry import list_families
from next.graph.catalog import build_catalog
from next.graph.node import Environment
from next.graph.registry import NodeRegistry
from next.graph.resolve import diagnose_node, resolve_node
from next.graph.runtime import GraphRunner
from next.starters import AnimaLoraOptions, SdxlLoraOptions, anima_lora, sdxl_lora

__all__ = ["AnimaLoraOptions", "DatasetStore", "Environment", "GraphRunner", "ModelLibrary", "NodeRegistry", "SdxlLoraOptions", "anima_lora",
           "build_catalog", "comfy_folders", "diagnose_node", "library_report", "list_families", "make_runtime", "resolve_node", "sdxl_lora"]


def make_runtime(context):
    """The run-wide residency and determinism runtime. Imported on the first run: the server never loads torch."""
    from next.engine.runtime.factory import make_runtime as make  # noqa: PLC0415

    return make(context)
