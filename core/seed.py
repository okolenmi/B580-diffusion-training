"""Deterministic seed derivation for reproducible randomness."""

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
