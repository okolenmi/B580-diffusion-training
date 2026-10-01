"""Deterministic seed derivation for reproducible randomness.

**Moved here from `core/seed.py` on 2026-10-02.** 18 lines, one
function, no dependencies beyond `hashlib`. It moved because
`nodes/model/unet_wrapper.py`'s `make_rand_cond()` needs it, and having
the UNet wrapper depend on `core/` purely for a SHA-256 call would have
kept the package boundary exactly where the point of this migration was
to move it. `core/seed.py` stays as a re-export shim, since
`manager/builder.py` and both of `core/`'s cache builders use it too.

Placed under `components/` rather than `model/`: this is not about any
one model. It is the rule that a given (base_seed, step, role) triple
always names the same randomness, and four unrelated call sites depend
on that being one shared function rather than four copies.
"""

import hashlib


def derive_seed(base: int, step: int, role: str) -> int:
    """
    Derive a reproducible 32-bit seed from (base_seed, global_step, role).

    Any code that needs randomness tied to a specific training step can call
    this independently — teacher cache, student forward, make_rand_cond, future
    augmentations — and will always agree on the value for the same inputs.

    'role' is a free-form string that namespaces the seed so different uses at
    the same step never collide (e.g. "x0", "noise", "cond").
    """
    key = f"{base}:{step}:{role}".encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**32)