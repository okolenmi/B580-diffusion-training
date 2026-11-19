"""Where the graph part of the backend takes the trainer's packages (``graph``, ``families``, ``data``, ``engine``,
``starters``) from, so the adapters beside it import this module instead of each reaching into the trainer.
"""

from __future__ import annotations

from data.library import report as library_report
from data.store import DatasetStore
from families.api.library import ModelLibrary, comfy_folders
from families.registry import list_families
from graph.catalog import build_catalog
from graph.node import Environment
from graph.registry import NodeRegistry
from graph.resolve import diagnose_node, resolve_node
from graph.runtime import GraphRunner
from starters import AnimaLoraOptions, SdxlImageOptions, SdxlLoraOptions, anima_lora, sdxl_image, sdxl_lora

__all__ = ["AnimaLoraOptions", "DatasetStore", "Environment", "GraphRunner", "ModelLibrary", "NodeRegistry", "SdxlImageOptions", "SdxlLoraOptions", "anima_lora",
           "build_catalog", "comfy_folders", "diagnose_node", "library_report", "list_families", "make_runtime", "resolve_node", "sdxl_image", "sdxl_lora"]


def make_runtime(context):
    """The run-wide residency and determinism runtime. Imported on the first run: the server never loads torch."""
    from engine.runtime.factory import make_runtime as make  # noqa: PLC0415

    return make(context)
