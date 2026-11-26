"""Loaded by every interpreter whose PYTHONPATH holds this folder (the browser tests' server and the graph children it starts).

With ``B580_TINY_PROFILE`` set, the model families answer with the tiny CPU stand-ins of ``tests/pipeline_nodes.py`` and the node library gains
their source nodes, so a real server runs real graphs (training, generation) in seconds with no weights. Without the variable it does nothing.
"""

import os
import sys

if os.environ.get("B580_TINY_PROFILE"):
    try:
        import importlib

        from graph.registry import NodeRegistry
        from tests.pipeline_nodes import TinyAnimaFamily, TinyFamily

        _default = NodeRegistry.default.__func__

        def _with_stand_ins(cls):
            registry = _default(cls)
            return cls(packages=(*registry.packages, "tests.pipeline_nodes"))

        NodeRegistry.default = classmethod(_with_stand_ins)
        importlib.import_module("families.sdxl.family").FAMILY = TinyFamily()
        importlib.import_module("families.anima.family").FAMILY = TinyAnimaFamily()

        # The memory gate refuses every run when no card reports a total, so the profile presents a card of this size (MB). Graphs of the
        # profile run on the CPU, the card exists only for the server's own bookkeeping.
        import torch
        from types import SimpleNamespace

        _mb = int(os.environ.get("B580_TINY_CARD_MB", "12000")) * 1024 * 1024
        xpu = torch.xpu
        xpu.is_available = lambda: True
        xpu.device_count = lambda: 1
        xpu.current_device = lambda: 0
        xpu.get_device_properties = lambda _i=0: SimpleNamespace(name="Tiny test card", total_memory=_mb)
        xpu.mem_get_info = lambda _i=None: (_mb, _mb)
        xpu.empty_cache = lambda: None
    except Exception as error:  # noqa: BLE001 - a profile that did not load must be loud, not a server that quietly runs the real thing
        print(f"tiny profile NOT loaded: {type(error).__name__}: {error}", file=sys.stderr)
