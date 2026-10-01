"""Compatibility shim: `derive_seed` now lives in
`nodes/components/seed.py`.

Moved there on 2026-10-02 because `nodes/model/unet_wrapper.py`'s
`make_rand_cond()` needs it, and a SHA-256 helper is not worth a
cross-package dependency on `core/`. `manager/builder.py` and both of
`core/`'s cache builders still call this; they keep working unchanged.
Retire this file when `core/` goes.
"""

from nodes.components.seed import derive_seed

__all__ = ["derive_seed"]