"""Import-boundary check for the trainer tree (``accept data engine families graph lab nodes perf tests`` plus
``console.py``, ``run.py`` and ``starters.py`` at the repository root).

    python scripts/check_imports.py            # exit 1 on any violation
    python scripts/check_imports.py --list     # print the layer table

Rules come from docs/developing.md. Each file belongs to one layer, decided by its path; a layer may
import only the layers listed for it, and a *pure* layer may import nothing outside the standard library. Inside one
top-level package imports are relative; between packages they are absolute. A relative import that climbs out of its
package is refused, and so is an import of the backend, the old trees or the root helpers the backend uses.

ComfyUI has top-level modules whose names ours share (``nodes.py``, ``utils``, ``server``, ...). Production code never imports
ComfyUI: only the two scripts that run as separate processes do, and each must take the repository root off ``sys.path``
before putting ComfyUI first. Both rules are checked here.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("accept", "data", "engine", "families", "graph", "lab", "nodes", "perf", "tests")
MODULES = ("console", "run", "starters")
NOT_OURS = {"manager", "backend", "core", "frontend", "scripts", "paths"}
#: The only files that import ComfyUI, and what they must do to keep our folders from shadowing it.
COMFY_SCRIPTS = {"accept/comfy_forward.py", "accept/comfy_render.py"}
COMFY_GUARD = "os.path.dirname(here)"

# layer -> (allowed internal layers, pure?)
LAYERS: dict[str, tuple[set[str], bool]] = {
    "graph": ({"graph"}, True),
    "families.api": ({"families.api", "graph"}, True),
    "families.registry": ({"families.api", "families.registry", "graph"}, True),
    "families.general": ({"families.api", "families.general", "graph"}, False),
    "families.<name>": ({"families.api", "families.general", "families.<own>", "graph"}, False),
    "data": ({"data", "graph", "families.api", "families.registry"}, False),
    "engine": ({"engine", "graph", "families.api", "families.registry", "data"}, False),
    "perf": ({"perf", "graph", "families.api", "families.registry", "families.general", "data", "engine"}, False),
    "accept": ({"accept", "perf", "graph", "families.<name>", "families.api", "families.registry", "families.general", "data", "engine"}, False),
    "nodes": ({"nodes", "perf", "graph", "families.api", "families.registry", "data", "engine"}, False),
    "tests": (set(), False),  # may import anything in the trainer tree
}


def layer_of_parts(parts: tuple[str, ...]) -> tuple[str, str | None] | None:
    """(layer, family name or None) for module parts starting at the top-level package, or None when not a layered package."""
    if not parts:
        return None
    top = parts[0]
    if top == "families":
        if len(parts) < 2:
            return None
        if parts[1] in {"api", "general", "registry"}:
            return f"families.{parts[1]}", None
        return "families.<name>", parts[1]
    if top in LAYERS:
        return top, None
    return None


def layer_of(path: Path) -> tuple[str, str | None] | None:
    return layer_of_parts(path.relative_to(ROOT).with_suffix("").parts)


def resolve_relative(src: Path, level: int, module: str | None) -> tuple[str, ...]:
    base = list(src.relative_to(ROOT).with_suffix("").parts)
    base = base[:-1]
    if level - 1 > len(base) - 1:
        return ("..",)  # leaves the top-level package
    base = base[: len(base) - (level - 1)]
    return tuple(base + (module.split(".") if module else []))


def imports_of(path: Path):
    """(line, relative?, module parts) for every import; ``from . import x`` yields the submodule ``x``."""
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, False, tuple(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parts = resolve_relative(path, node.level, module)
                if not module and parts != ("..",):
                    for alias in node.names:
                        yield node.lineno, True, parts + (alias.name,)
                else:
                    yield node.lineno, True, parts
            else:
                yield node.lineno, False, tuple(module.split("."))


def check_file(path: Path) -> list[str]:
    info = layer_of(path)
    if info is None:
        return []
    layer, family = info
    allowed, pure = LAYERS[layer]
    problems: list[str] = []
    rel = path.relative_to(ROOT)
    for lineno, relative, parts in imports_of(path):
        if parts and parts[0] == "..":
            problems.append(f"{rel}:{lineno}: relative import leaves its package; import the other package by its name")
            continue
        top = parts[0]
        if top in NOT_OURS:
            problems.append(f"{rel}:{lineno}: imports {top}, which the trainer tree must not depend on")
            continue
        if top == "comfy" and rel.as_posix() not in COMFY_SCRIPTS:
            problems.append(f"{rel}:{lineno}: imports ComfyUI; only {', '.join(sorted(COMFY_SCRIPTS))} may")
            continue
        target = layer_of_parts(parts)
        if target is None:
            if pure and not relative and top not in sys.stdlib_module_names and top != "__future__" and top not in MODULES:
                problems.append(f"{rel}:{lineno}: pure layer {layer} imports non-stdlib module {top}")
            continue
        t_layer, t_family = target
        if layer == "tests" or (t_layer == layer and (layer != "families.<name>" or t_family == family)):
            continue
        if relative and t_layer != layer:
            continue  # a relative import cannot reach another package; the case above caught the climb
        ok = t_layer in allowed or (t_layer == "families.<name>" and "families.<own>" in allowed and t_family == family)
        if not ok:
            label = f"{t_layer}" + (f" ({t_family})" if t_family else "")
            problems.append(f"{rel}:{lineno}: layer {layer} may not import {label}")
    return problems


def check_comfy_guard() -> list[str]:
    problems = []
    for name in sorted(COMFY_SCRIPTS):
        path = ROOT / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif COMFY_GUARD not in path.read_text(encoding="utf8"):
            problems.append(f"{name}: must take the repository root off sys.path ({COMFY_GUARD}) before ComfyUI goes first")
    return problems


def tracked_files() -> list[Path]:
    files = [ROOT / f"{m}.py" for m in MODULES if (ROOT / f"{m}.py").exists()]
    for package in PACKAGES:
        files += (p for p in (ROOT / package).rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


def main(argv: list[str]) -> int:
    if "--list" in argv:
        for name, (allowed, pure) in LAYERS.items():
            print(f"{name:20} pure={pure!s:5} may import: {', '.join(sorted(allowed)) or 'anything in the trainer tree'}")
        return 0
    files = tracked_files()
    problems: list[str] = []
    for path in files:
        problems.extend(check_file(path))
    problems.extend(check_comfy_guard())
    for line in problems:
        print(line)
    print(f"check_imports: {len(files)} files, {len(problems)} violation(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
