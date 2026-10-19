"""How much a node asks of the person looking at it.

    python -m next.tests.check_node_surface

A node with many fields stays readable only when they are grouped: every real node with more than FLAT_LIMIT
parameters must put all of them in groups, no group may exceed GROUP_LIMIT, and what is visible without opening
anything (groups neither collapsed nor advanced) stays within FIRST_VIEW. New options go in a collapsed group.
"""

from __future__ import annotations

from ..graph import NodeRegistry
from .support import check, run

FLAT_LIMIT, GROUP_LIMIT, FIRST_VIEW = 6, 9, 9
PLAIN = (int, float, str, bool)


def surface():
    classes, _ = NodeRegistry.default().load()
    flat, crowded, first = [], [], []
    for name, cls in sorted(classes.items()):
        params = {k: p for k, p in cls.INPUTS.items() if p.type in PLAIN}
        groups = {g.id: g for g in cls.GROUPS}
        if len(params) > FLAT_LIMIT and any(not p.group for p in params.values()):
            flat.append(name)
        sizes: dict[str, int] = {}
        for p in params.values():
            if p.group:
                sizes[p.group] = sizes.get(p.group, 0) + 1
        crowded += [f"{name}.{g}" for g, n in sizes.items() if n > GROUP_LIMIT]
        shown = sum(n for g, n in sizes.items() if g in groups and not groups[g].collapsed and not groups[g].advanced)
        shown += sum(1 for p in params.values() if not p.group)
        if shown > FIRST_VIEW:
            first.append(f"{name} ({shown})")
    check(not flat, f"nodes with more than {FLAT_LIMIT} parameters group all of them (ungrouped: {flat})")
    check(not crowded, f"no group holds more than {GROUP_LIMIT} fields ({crowded})")
    check(not first, f"at most {FIRST_VIEW} fields are visible before anything is opened ({first})")


if __name__ == "__main__":
    run(surface)
