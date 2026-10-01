"""Compatibility shim: `SDXLClipEncoder` now lives in
`nodes/model/clip_encoder.py`.

Text encoding was unwired from `core/` on 2026-10-02 -- the node graph's
conditioning path had been running through this module, and the
implementation was self-contained enough to relocate rather than
reimplement. `nodes/model/` holds the real code; this file only
re-exports it, so existing callers inside `core/` (`trainer.py`, both
cache builders) and `manager/builder.py` keep working unchanged.

Kept rather than deleted because `core/` is still the production
training path -- the backend spawns `python -m core.cli`, which runs
`core/trainer.py`, which builds this encoder. Retire this shim when
`core/` goes; there is no behavior here worth preserving.
"""

from nodes.model.clip_encoder import SDXLClipEncoder, _extract_and_convert_clip_state_dict

__all__ = ["SDXLClipEncoder", "_extract_and_convert_clip_state_dict"]