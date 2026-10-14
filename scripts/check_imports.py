"""Import-boundary check for the rework tree (``next/``).

    python scripts/check_imports.py            # exit 1 on any violation
    python scripts/check_imports.py --list     # print the layer table

Rules come from rework/80-developer-guide.md section 2.1. Each file under
``next/`` belongs to one layer, decided by its path; a layer may import only the
layers listed for it, and a *pure* layer may import nothing outside the standard
library. Relative imports are resolved before checking, so a ``..`` that climbs
into a forbidden layer is caught the same as an absolute import.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "next"
OLD_TREES = {"nodes", "manager", "backend", "core", "frontend", "scripts", "monitor_bus", "paths"}

# layer -> (allowed internal layers, pure?)
LAYERS: dict[str, tuple[set[str], bool]] = {
    "graph": ({"graph"}, True),
    "families.api": ({"families.api", "graph"}, True),
    "families.registry": ({"families.api", "families.registry", "graph"}, True),
    "families.general": ({"families.api", "families.general", "graph"}, False),
    "families.<name>": ({"families.api", "families.general", "families.<own>", "graph"}, False),
    "data": ({"data", "graph", "families.api", "families.registry"}, False),
    "engine": ({"engine", "graph", "families.api", "families.registry", "data"}, False),
    "nodes": ({"nodes", "graph", "families.api", "families.registry", "data", "engine"}, False),
    "tests": (set(), False),  # may import anything inside next/
}


def layer_of(path: Path) -> tuple[str, str | None] | None:
    """(layer, family name or None) for a file under next/, or None for the root."""
    parts = path.relative_to(TREE).with_suffix("").parts
    if not parts:
        return None
    top = parts[0]
    if top == "families":
        if len(parts) < 2:
            return None
        second = parts[1]
        if second in {"api", "general"}:
            return f"families.{second}", None
        if second == "registry":
            return "families.registry", None
        return "families.<name>", second
    if top in {"graph", "data", "engine", "nodes", "tests"}:
        return top, None
    return None


def target_layer(parts: tuple[str, ...]) -> tuple[str, str | None] | None:
    """Layer of an import target given as parts relative to next/."""
    if not parts:
        return None
    if parts[0] == "families":
        if len(parts) < 2:
            return None
        if parts[1] in {"api", "general", "registry"}:
            return f"families.{parts[1]}", None
        return "families.<name>", parts[1]
    if parts[0] in {"graph", "data", "engine", "nodes", "tests"}:
        return parts[0], None
    return None


def resolve_relative(src: Path, level: int, module: str | None) -> tuple[str, ...]:
    base = list(src.relative_to(TREE).with_suffix("").parts)
    if src.name != "__init__.py":
        base = base[:-1]
    else:
        base = base[:-1] if base and base[-1] == "__init__" else base
    if level - 1 > len(base):
        return ("..",)
    base = base[: len(base) - (level - 1)]
    return tuple(base + (module.split(".") if module else []))


def imports_of(path: Path):
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, 0, tuple(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parts = resolve_relative(path, node.level, module)
                names = [a.name for a in node.names]
                # `from . import x` imports submodule x; treat each name as a path part
                if not module:
                    for name in names:
                        yield node.lineno, node.level, parts + (name,)
                else:
                    yield node.lineno, node.level, parts
            else:
                yield node.lineno, 0, tuple(module.split("."))


def check_file(path: Path) -> list[str]:
    info = layer_of(path)
    if info is None:
        return []
    layer, family = info
    allowed, pure = LAYERS[layer]
    problems: list[str] = []
    rel = path.relative_to(ROOT)
    for lineno, level, parts in imports_of(path):
        if parts and parts[0] == "..":
            problems.append(f"{rel}:{lineno}: relative import leaves next/")
            continue
        if level:
            target = target_layer(parts)
            if target is None:
                continue  # inside the same package root or an unknown helper module
            t_layer, t_family = target
            if layer == "tests":
                continue
            ok = t_layer in allowed or (
                t_layer == "families.<name>" and "families.<own>" in allowed and t_family == family
            )
            # A layer always may import itself.
            if t_layer == layer and (layer != "families.<name>" or t_family == family):
                ok = True
            if not ok:
                label = f"{t_layer}" + (f" ({t_family})" if t_family else "")
                problems.append(f"{rel}:{lineno}: layer {layer} may not import {label}")
        else:
            top = parts[0]
            if top == "next":
                problems.append(f"{rel}:{lineno}: absolute import of next.*; use a relative import")
            elif top in OLD_TREES:
                problems.append(f"{rel}:{lineno}: imports the old tree ({top})")
            elif pure and top not in sys.stdlib_module_names and top != "__future__":
                problems.append(f"{rel}:{lineno}: pure layer {layer} imports non-stdlib module {top}")
    return problems


def main(argv: list[str]) -> int:
    if "--list" in argv:
        for name, (allowed, pure) in LAYERS.items():
            print(f"{name:20} pure={pure!s:5} may import: {', '.join(sorted(allowed)) or 'anything in next/'}")
        return 0
    if not TREE.exists():
        print("next/ does not exist; nothing to check")
        return 0
    problems: list[str] = []
    files = sorted(p for p in TREE.rglob("*.py") if "__pycache__" not in p.parts)
    for path in files:
        problems.extend(check_file(path))
    for line in problems:
        print(line)
    print(f"check_imports: {len(files)} files, {len(problems)} violation(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
