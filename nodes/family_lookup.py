"""The family object behind a model part.

An environment may supply its own ``families`` service (a callable taking the family id); a stand-in family in a test
goes in that way. Without one the registry answers, which is what a real run does.
"""

from __future__ import annotations

from typing import Any

from families.registry import get_family


def family_of(context: Any, family_id: str) -> Any:
    env = context.env
    if env.has("families"):
        return env.service("families")(family_id)
    return get_family(family_id)
